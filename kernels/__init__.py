from kernels.affine_qmv import affine_qmv, affine_swiglu_qmv
from kernels.attention import (
    ATTENTION_DECODE_CONFIGS,
    ATTENTION_PARTIAL_CONFIGS,
)
from kernels.gemm import MATMUL_CONFIGS, matmul, matmul_relu, matmul_swizzled
from kernels.layernorm import layernorm
from kernels.mlp import matmul_gelu, matmul_silu
from kernels.reduce import REDUCE_KERNELS, reduce_2, reduce_4, reduce_8, reduce_16
from kernels.rmsnorm import rmsnorm
from kernels.simdgroup_specialized_elementwise import (
    exp_kernel,
    exp_sqrt_kernel,
    geglu_kernel,
    geglu_specialized_kernel,
    gelu_kernel,
    gelu_silu_kernel,
    silu_kernel,
    sqrt_abs_kernel,
)
from kernels.softmax import softmax


def __getattr__(name):
    if name == "attention_decode":
        from metile.runtime.attention import attention_decode

        globals()[name] = attention_decode
        return attention_decode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ATTENTION_DECODE_CONFIGS",
    "ATTENTION_PARTIAL_CONFIGS",
    "MATMUL_CONFIGS",
    "REDUCE_KERNELS",
    "affine_qmv",
    "affine_swiglu_qmv",
    "attention_decode",
    "exp_kernel",
    "exp_sqrt_kernel",
    "geglu_kernel",
    "geglu_specialized_kernel",
    "gelu_kernel",
    "gelu_silu_kernel",
    "layernorm",
    "matmul",
    "matmul_gelu",
    "matmul_relu",
    "matmul_silu",
    "matmul_swizzled",
    "reduce_2",
    "reduce_4",
    "reduce_8",
    "reduce_16",
    "rmsnorm",
    "silu_kernel",
    "softmax",
    "sqrt_abs_kernel",
]
