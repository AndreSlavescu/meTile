"""First-order dense GEMM training composed from existing DSL matrix kernels.

All products use FP32 operands/accumulation and an explicit SIMD-group schedule
instead of an automatically selected reduced-precision tensor-ops backend.
Packing and transposes are GPU kernels, not NumPy computations. This bounded
baseline favors a checked numerical contract over a throughput claim.
Only contiguous logical 2-D A[M,K] @ B[K,N] is supported. No bias, batching,
quantized weights, in-place aliasing or implicit framework registration.
"""

from dataclasses import dataclass
from math import prod

import numpy as np

from metile.backends.pointwise import (
    ActivationContext,
    _array,
    _buffer,
    activation_backward,
    activation_forward,
)
from metile.compiler.options import Schedule
from metile.runtime.buffer import MtileBuffer

DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024


def _budget(elements, limit):
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_workspace_bytes must be a positive integer")
    if elements * 4 > limit:
        raise ValueError(f"training matmul requires up to {elements * 4} workspace bytes")


def _pack(values, *, transpose=False, dtype=np.float32):
    from metile_kernels.matrix_pack import matrix_pack

    shape = values.shape[::-1] if transpose else values.shape
    output = MtileBuffer.empty(shape, dtype)
    matrix_pack[((prod(shape) + 127) // 128,)](
        values, output, *values.shape, TRANSPOSE=transpose, BLOCK=128, STRICT_MATH=True
    )
    return output


def _multiply(left, right):
    from metile_kernels.gemm import matmul

    rows, reduction = left.shape
    columns = right.shape[1]
    output = MtileBuffer.empty((rows, columns), np.float32)
    matmul.kernel_fn[((rows + 31) // 32, (columns + 31) // 32)](
        left,
        right,
        output,
        rows,
        columns,
        reduction,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        SCHEDULE=Schedule(backend="simdgroup"),
        STRICT_MATH=True,
    )
    return output


@dataclass(frozen=True)
class MatmulContext:
    left: MtileBuffer
    right: MtileBuffer
    activation: ActivationContext | None


def matmul_forward(left, right, *, activation=None, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Return (FP32 output, context), optionally with a named activation.

    ``quick_gelu`` matches the legacy matmul_gelu approximation; ``gelu_tanh``
    is a different operation. Saved packed inputs are independent of caller
    buffers. The saved activation input must not be mutated before backward.
    """
    from metile_kernels.training_activations import _validate

    _array(left, "left")
    _array(right, "right")
    if len(left.shape) != 2 or len(right.shape) != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("matmul requires A[M,K] and B[K,N]")
    if activation is not None:
        _validate(activation, False, 128)
    rows, columns = left.shape[0], right.shape[1]
    if rows * columns >= 2**31:
        raise ValueError("output exceeds signed-32-bit indexing")
    _budget(
        prod(left.shape) + prod(right.shape) + rows * columns * (2 if activation else 1),
        max_workspace_bytes,
    )
    packed_left, packed_right = _pack(_buffer(left)), _pack(_buffer(right))
    output = _multiply(packed_left, packed_right)
    activation_context = None
    if activation is not None:
        output, activation_context = activation_forward(output, kind=activation)
    return output, MatmulContext(packed_left, packed_right, activation_context)


def matmul_backward(context, gradient, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Return FP32 dA=dOutput@B.T and dB=A.T@dOutput, without atomics."""
    if not isinstance(context, MatmulContext):
        raise TypeError("matmul backward requires a MatmulContext")
    shape = (context.left.shape[0], context.right.shape[1])
    _array(gradient, "gradient", shape)
    _budget(
        2 * (prod(context.left.shape) + prod(context.right.shape)) + 2 * prod(shape),
        max_workspace_bytes,
    )
    gradient = _pack(_buffer(gradient))
    if context.activation is not None:
        gradient = activation_backward(context.activation, gradient)
    right_transposed = _pack(context.right, transpose=True)
    left_transposed = _pack(context.left, transpose=True)
    return _multiply(gradient, right_transposed), _multiply(left_transposed, gradient)
