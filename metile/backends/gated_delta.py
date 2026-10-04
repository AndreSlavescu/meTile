"""Functional Gated DeltaNet/KDA recurrence and explicit first-order VJP.

Inputs are query/key [B,T,H,K], value [B,T,H,V], beta [B,T,H], and
log_decay [B,T,H] (scalar) or [B,T,H,K] (channel). State is [B,H,K,V].
This BTHD convention differs from the BHSD softmax-attention interface.
All heads match: grouped heads must be expanded by the caller. Normalization,
short convolution, gate activations, masks, cache mutation, and model adapters
are deliberately outside this operator. Scale is fixed, not differentiated.

Each token computes U = exp(log_decay) * state, error = value - U.T @ key,
state = U + beta * outer(key, error), output = scale * state.T @ query.
This is the decay-before-correction convention of Kimi Linear equation (1):
https://arxiv.org/html/2510.26692v1 . The scalar-decay special case is used by
Qwen3-Next: https://github.com/huggingface/transformers/blob/main/src/transformers/
models/qwen3_next/modeling_qwen3_next.py . No model compatibility is implied.

Native kernels require contiguous FP32 inputs, B <= 64, T <= 4096, H <= 128,
and K,V <= 256. Finite inputs are a precondition, not scanned by native dispatch.
They use strict arithmetic and deterministic partial reductions, not atomics.
Saved states cost 4*B*(T+1)*H*K*V bytes. Backward additionally allocates three
B*T*H*V*K partial tensors and one B*T*H*V partial tensor. The allocation budget
covers newly allocated output/workspace buffers, not caller-owned inputs or
copies of NumPy inputs. No performance or framework-autograd claim is made.
"""

from dataclasses import dataclass
from math import isfinite, prod

import numpy as np

from metile.runtime.buffer import MtileBuffer

DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class GatedDeltaResult:
    output: object
    final_state: object
    states: object | None = None


@dataclass(frozen=True)
class GatedDeltaGradients:
    query: object
    key: object
    value: object
    log_decay: object
    beta: object
    initial_state: object


def _array(name, value, shape, dtype):
    if not isinstance(value, (np.ndarray, MtileBuffer)):
        raise TypeError(f"{name} must be a NumPy array or meTile Buffer")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} expects shape {shape}, received {value.shape}")
    if np.dtype(value.dtype) != dtype:
        raise TypeError(f"{name} expects {dtype}, received {value.dtype}")
    if isinstance(value, np.ndarray) and not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if prod(shape) >= 2**31:
        raise ValueError(f"{name} exceeds the signed 32-bit indexing limit")


def _dimensions(query, key, value, log_decay, beta, scale, *, reference=False):
    if not isinstance(query, (np.ndarray, MtileBuffer)) or len(query.shape) != 4:
        raise ValueError("query must have shape [B,T,H,K]")
    batch, sequence, heads, key_dim = query.shape
    if not isinstance(value, (np.ndarray, MtileBuffer)) or len(value.shape) != 4:
        raise ValueError("value must have shape [B,T,H,V]")
    value_dim = value.shape[-1]
    if any(dimension <= 0 for dimension in (batch, sequence, heads, key_dim, value_dim)):
        raise ValueError("all gated-delta dimensions must be positive")
    if batch > 64 or sequence > 4096 or heads > 128 or key_dim > 256 or value_dim > 256:
        raise ValueError("gated delta supports B <= 64, T <= 4096, H <= 128 and K,V <= 256")
    dtype = np.dtype(query.dtype)
    allowed = (np.dtype("float32"), np.dtype("float64")) if reference else (np.dtype("float32"),)
    if dtype not in allowed:
        raise TypeError("gated delta requires FP32 inputs (FP64 is reference-only)")
    if (
        not isinstance(scale, (int, float))
        or not isfinite(scale)
        or abs(scale) > float(np.finfo(dtype).max)
    ):
        raise ValueError("scale must be finite and representable in the input dtype")
    prefix = (batch, sequence, heads)
    _array("query", query, (*prefix, key_dim), dtype)
    _array("key", key, (*prefix, key_dim), dtype)
    _array("value", value, (*prefix, value_dim), dtype)
    _array("beta", beta, prefix, dtype)
    channel_decay = getattr(log_decay, "shape", None) == (*prefix, key_dim)
    _array("log_decay", log_decay, (*prefix, key_dim) if channel_decay else prefix, dtype)
    return batch, sequence, heads, key_dim, value_dim, channel_decay, dtype


def _budget(shapes, limit):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("max_workspace_bytes must be a positive integer")
    for shape in shapes:
        if prod(shape) >= 2**31:
            raise ValueError("workspace exceeds the signed 32-bit indexing limit")
    required = 4 * sum(prod(shape) for shape in shapes)
    if required > limit:
        raise ValueError(f"gated delta needs {required} workspace/output bytes; budget is {limit}")


def _buffer(value):
    return value if isinstance(value, MtileBuffer) else MtileBuffer.from_numpy(value)


def gated_delta_forward(
    query,
    key,
    value,
    log_decay,
    beta,
    initial_state,
    *,
    scale=1.0,
    save_states=False,
    max_workspace_bytes=DEFAULT_WORKSPACE_BYTES,
):
    """Run FP32 Metal recurrence; optional states include the initial state at t=0."""
    from metile_kernels.gated_delta import gated_delta_forward_kernel

    batch, sequence, heads, key_dim, value_dim, channel, dtype = _dimensions(
        query, key, value, log_decay, beta, scale
    )
    state_shape = (batch, heads, key_dim, value_dim)
    states_shape = (batch, sequence + 1, heads, key_dim, value_dim)
    _array("initial_state", initial_state, state_shape, dtype)
    shapes = [value.shape, state_shape, states_shape if save_states else (1,)]
    _budget(shapes, max_workspace_bytes)
    output, final_state, states = [MtileBuffer.empty(shape) for shape in shapes]
    inputs = [_buffer(array) for array in (query, key, value, log_decay, beta, initial_state)]
    gated_delta_forward_kernel[(batch * heads * value_dim,)](
        *inputs,
        output,
        final_state,
        states,
        sequence,
        float(scale),
        BATCH=batch,
        HEADS=heads,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        CHANNEL_DECAY=channel,
        SAVE_STATES=bool(save_states),
        BLOCK=max(32, 1 << (key_dim - 1).bit_length()),
        STRICT_MATH=True,
    )
    return GatedDeltaResult(output, final_state, states if save_states else None)


def gated_delta_backward(
    query,
    key,
    value,
    log_decay,
    beta,
    states,
    output_gradient,
    final_state_gradient,
    *,
    scale=1.0,
    max_workspace_bytes=DEFAULT_WORKSPACE_BYTES,
):
    """Explicit VJP using states from the same forward inputs and fixed scale.

    Both output cotangents are required; pass zeros for an unused final state.
    Inputs and saved states are read-only. This does not register framework AD.
    """
    from metile_kernels.gated_delta import (
        gated_delta_backward_kernel,
        gated_delta_reduce_gradients_kernel,
    )

    batch, sequence, heads, key_dim, value_dim, channel, dtype = _dimensions(
        query, key, value, log_decay, beta, scale
    )
    state_shape = (batch, heads, key_dim, value_dim)
    _array("states", states, (batch, sequence + 1, heads, key_dim, value_dim), dtype)
    _array("output_gradient", output_gradient, value.shape, dtype)
    _array("final_state_gradient", final_state_gradient, state_shape, dtype)
    partial_shape = (batch, sequence, heads, value_dim, key_dim)
    gradient_shapes = [
        query.shape,
        key.shape,
        value.shape,
        log_decay.shape,
        beta.shape,
        state_shape,
    ]
    partial_shapes = [partial_shape, partial_shape, partial_shape, value.shape]
    _budget(gradient_shapes + partial_shapes, max_workspace_bytes)
    gradients = [MtileBuffer.empty(shape) for shape in gradient_shapes]
    partials = [MtileBuffer.empty(shape) for shape in partial_shapes]
    inputs = [
        _buffer(array)
        for array in (
            query,
            key,
            value,
            log_decay,
            beta,
            states,
            output_gradient,
            final_state_gradient,
        )
    ]
    gated_delta_backward_kernel[(batch * heads * value_dim,)](
        *inputs,
        *partials,
        gradients[2],
        gradients[5],
        sequence,
        float(scale),
        BATCH=batch,
        HEADS=heads,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        CHANNEL_DECAY=channel,
        BLOCK=max(32, 1 << (key_dim - 1).bit_length()),
        STRICT_MATH=True,
    )
    gated_delta_reduce_gradients_kernel[(batch * sequence * heads,)](
        *partials,
        gradients[0],
        gradients[1],
        gradients[3],
        gradients[4],
        ROWS=batch * sequence * heads,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        CHANNEL_DECAY=channel,
        BLOCK=max(32, 1 << (max(key_dim, value_dim) - 1).bit_length()),
        STRICT_MATH=True,
    )
    return GatedDeltaGradients(*gradients)


def _reference_inputs(arrays):
    if any(not isinstance(array, np.ndarray) for array in arrays):
        raise TypeError("the CPU reference accepts only NumPy arrays")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("the CPU reference requires finite inputs")


def gated_delta_reference(query, key, value, log_decay, beta, initial_state, *, scale=1.0):
    """NumPy FP32/FP64 oracle, always saving every state; no Metal initialization."""
    _reference_inputs((query, key, value, log_decay, beta, initial_state))
    batch, sequence, heads, key_dim, value_dim, channel, dtype = _dimensions(
        query, key, value, log_decay, beta, scale, reference=True
    )
    _array("initial_state", initial_state, (batch, heads, key_dim, value_dim), dtype)
    state = initial_state.copy()
    states = np.empty((batch, sequence + 1, heads, key_dim, value_dim), dtype=dtype)
    states[:, 0] = state
    output = np.empty_like(value)
    for token in range(sequence):
        decay = np.exp(log_decay[:, token])
        decay = decay[..., None] if channel else decay[..., None, None]
        decayed = state * decay
        error = value[:, token] - np.einsum("bhkv,bhk->bhv", decayed, key[:, token])
        state = (
            decayed
            + beta[:, token, :, None, None] * key[:, token, :, :, None] * error[..., None, :]
        )
        output[:, token] = scale * np.einsum("bhkv,bhk->bhv", state, query[:, token])
        states[:, token + 1] = state
    return GatedDeltaResult(output, state, states)


def gated_delta_reference_backward(
    query,
    key,
    value,
    log_decay,
    beta,
    states,
    output_gradient,
    final_state_gradient,
    *,
    scale=1.0,
):
    """NumPy reverse recurrence, including initial/final-state cotangents."""
    _reference_inputs(
        (query, key, value, log_decay, beta, states, output_gradient, final_state_gradient)
    )
    batch, sequence, heads, key_dim, value_dim, channel, dtype = _dimensions(
        query, key, value, log_decay, beta, scale, reference=True
    )
    _array("states", states, (batch, sequence + 1, heads, key_dim, value_dim), dtype)
    _array("output_gradient", output_gradient, value.shape, dtype)
    _array("final_state_gradient", final_state_gradient, (batch, heads, key_dim, value_dim), dtype)
    query_gradient, key_gradient, value_gradient, decay_gradient, beta_gradient = [
        np.empty_like(array) for array in (query, key, value, log_decay, beta)
    ]
    state_gradient = final_state_gradient.copy()
    for token in reversed(range(sequence)):
        query_token, key_token = query[:, token], key[:, token]
        output_token = output_gradient[:, token]
        decay = np.exp(log_decay[:, token])
        decay = decay[..., None] if channel else decay[..., None, None]
        decayed = states[:, token] * decay
        error = value[:, token] - np.einsum("bhkv,bhk->bhv", decayed, key_token)
        total = state_gradient + scale * query_token[..., :, None] * output_token[..., None, :]
        projected = np.einsum("bhkv,bhk->bhv", total, key_token)
        error_gradient = beta[:, token, :, None] * projected
        decayed_gradient = total - key_token[..., :, None] * error_gradient[..., None, :]
        query_gradient[:, token] = scale * np.einsum(
            "bhkv,bhv->bhk", states[:, token + 1], output_token
        )
        key_gradient[:, token] = beta[:, token, :, None] * np.einsum(
            "bhkv,bhv->bhk", total, error
        ) - np.einsum("bhkv,bhv->bhk", decayed, error_gradient)
        value_gradient[:, token] = error_gradient
        beta_gradient[:, token] = np.sum(projected * error, axis=-1)
        decay_product = decayed_gradient * decayed
        decay_gradient[:, token] = np.sum(decay_product, axis=-1 if channel else (-2, -1))
        state_gradient = decayed_gradient * decay
    return GatedDeltaGradients(
        query_gradient, key_gradient, value_gradient, decay_gradient, beta_gradient, state_gradient
    )
