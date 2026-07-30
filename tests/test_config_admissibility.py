"""Configurations a device cannot host must be pruned, not allowed to fail the whole shape.

A tuner offers several block sizes and some of them ask for more threadgroup memory than the part
has. That is a fact about one candidate, so the tuner should drop it and try the rest. Getting this
wrong is quiet and expensive: head dimension 256 works at five of the six block sizes the attention
tuner offers and reaches the limit only at 1024, and because one candidate raised, attention on
Qwen3-VL fell back to MLX for every shape and was recorded as unsupported.
"""

import pytest

from metile.frontend.kernel import OutOfResources


def test_out_of_resources_is_a_runtime_error_subclass():
    """Existing callers catch RuntimeError, and narrowing the type must not break them."""
    assert issubclass(OutOfResources, RuntimeError)


def test_only_resource_exhaustion_is_pruned():
    """Any other compile failure is a bug and has to surface.

    Catching RuntimeError broadly in a tuning loop would swallow both, which is why the exception is
    typed at all.
    """
    from metile.backends.mlx import _admissible

    assert _admissible(lambda: "compiled") == "compiled"
    assert _admissible(lambda: (_ for _ in ()).throw(OutOfResources("too big"))) is None
    for error in (RuntimeError("a real bug"), ValueError("wrong shape")):
        with pytest.raises(type(error)):
            _admissible(lambda error=error: (_ for _ in ()).throw(error))


def test_the_attention_kernel_reports_out_of_resources_rather_than_a_bare_error():
    """The kernel that motivated this: D=256 at BLOCK=1024 needs 40960 bytes against 32768."""
    import numpy as np

    import metile
    from kernels.attention import attention_decode_kernel

    heads, tokens, dimension = 2, 32, 256
    rng = np.random.default_rng(0)
    with pytest.raises(OutOfResources, match="threadgroup memory"):
        attention_decode_kernel[(heads,)](
            metile.Buffer(data=rng.standard_normal(heads * dimension, dtype=np.float32)),
            metile.Buffer(data=rng.standard_normal(heads * tokens * dimension, dtype=np.float32)),
            metile.Buffer(data=rng.standard_normal(heads * tokens * dimension, dtype=np.float32)),
            metile.Buffer.zeros((heads * dimension,)),
            tokens,
            float(dimension**-0.5),
            D=dimension,
            Q_HEADS=heads,
            KV_HEADS=heads,
            BLOCK=1024,
        )


@pytest.mark.parametrize("dimension", (64, 128, 256))
def test_attention_decode_matches_mlx_at_every_head_dimension(dimension):
    """Including 256, which the tuner used to decline entirely."""
    mx = pytest.importorskip("mlx.core")

    from metile.backends.mlx import mlx_attention_decode

    scale = float(dimension**-0.5)
    query = mx.random.normal((1, 8, 1, dimension)).astype(mx.float16)
    keys = mx.random.normal((1, 8, 256, dimension)).astype(mx.float16)
    values = mx.random.normal((1, 8, 256, dimension)).astype(mx.float16)
    mx.eval(query, keys, values)

    got = mlx_attention_decode(query, keys, values, scale=scale)
    reference = mx.fast.scaled_dot_product_attention(query, keys, values, scale=scale)
    mx.eval(got, reference)
    assert mx.allclose(got, reference, rtol=2e-3, atol=2e-3).item()


def test_affine_nax_weights_reject_widths_the_matrix_unit_cannot_decode():
    """The 4-bit guard is load-bearing, not a formatting preference.

    `lower_affine_matmul` emits NAX affine fragments with block_size=4 and has no bit-width
    parameter, so an 8-bit weight loaded through here is decoded as nibbles: the kernel reads eight
    values per word where the data holds four, returns a relative error of 2.5 to 2.9, and runs
    1.6x to 2.0x faster than MLX precisely because it does half the work. A large speedup arriving
    with a wrong answer is one bug, not one win and one bug.

    Pinned as a test because relaxing the check looks harmless and the failure is silent: only the
    tuner's agreement gate stopped the wrong kernel being selected.
    """
    mx = pytest.importorskip("mlx.core")

    from metile.backends.mlx_affine import MLXAffineWeight

    dense = mx.random.normal((256, 128)).astype(mx.float16)
    for bits in (2, 8):
        packed, scales, biases = mx.quantize(dense, group_size=64, bits=bits, mode="affine")
        mx.eval(packed, scales, biases)
        with pytest.raises(ValueError, match="4 bits"):
            MLXAffineWeight.from_mlx(packed, scales, biases, group_size=64, bits=bits)

    packed, scales, biases = mx.quantize(dense, group_size=64, bits=4, mode="affine")
    mx.eval(packed, scales, biases)
    weight = MLXAffineWeight.from_mlx(packed, scales, biases, group_size=64, bits=4)
    assert weight.bits == 4
