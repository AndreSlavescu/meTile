"""Paired activation kernels with FP32 intermediates and FP32 gradients.

Gated SiLU and tanh-GELU implement SwiGLU and GeGLU, respectively. The
``quick_gelu`` option preserves the older sigmoid(1.702*x) approximation;
it is deliberately not called the same operation as tanh-GELU.
These are independent implementations of the mathematical operations also
provided by https://github.com/linkedin/Liger-Kernel.

Launch ceil(SIZE/BLOCK) groups, BLOCK a multiple of 32, STRICT_MATH=True.
Inputs/outputs are disjoint contiguous FP16/FP32 arrays; gradients are FP32.
Both operands of a gated activation have SIZE elements (no broadcasting).
Unused Up/GradUp pointers may alias a valid buffer when GATED=False.
ReLU chooses derivative zero at zero. Only finite input values are supported.
"""

import metile


def _activation(value, kind):
    if kind in ("silu", "sigmoid", "quick_gelu"):
        slope = 1.702 if kind == "quick_gelu" else 1.0
        argument = value * slope
        exponential = metile.exp(0.0 - metile.abs(argument))
        sigmoid = metile.where(
            argument >= 0.0, 1.0 / (1.0 + exponential), exponential / (1.0 + exponential)
        )
        if kind == "sigmoid":
            return sigmoid, sigmoid * (1.0 - sigmoid)
        return value * sigmoid, sigmoid + (value * sigmoid * (1.0 - sigmoid)) * slope
    if kind == "gelu_tanh":
        bounded = metile.minimum(metile.maximum(value, -10.0), 10.0)
        square = bounded * bounded
        tangent = metile.tanh(0.7978845608028654 * bounded * (1.0 + 0.044715 * square))
        activation = 0.5 * value * (1.0 + tangent)
        derivative = 0.5 * (1.0 + tangent) + (
            0.5
            * bounded
            * (1.0 - tangent * tangent)
            * 0.7978845608028654
            * (1.0 + 0.134145 * square)
        )
        return activation, derivative
    if kind == "relu":
        return metile.maximum(value, 0.0), metile.where(value > 0.0, 1.0, 0.0)
    if kind == "tanh":
        tangent = metile.tanh(value)
        return tangent, 1.0 - tangent * tangent
    raise ValueError("KIND must be silu, sigmoid, gelu_tanh, quick_gelu, relu or tanh")


def _validate(kind, gated, block):
    if kind not in ("silu", "sigmoid", "gelu_tanh", "quick_gelu", "relu", "tanh"):
        raise ValueError("unsupported activation KIND")
    if type(gated) is not bool:
        raise TypeError("GATED must be bool")
    if type(block) is not int or block < 32 or block > 1024 or block % 32:
        raise ValueError("BLOCK must be a multiple of 32 in [32, 1024]")


@metile.kernel
def activation_forward(
    Input,
    Up,
    Output,
    SIZE,
    *,
    KIND: metile.constexpr = "silu",
    GATED: metile.constexpr = False,
    BLOCK: metile.constexpr = 128,
):
    _validate(KIND, GATED, BLOCK)
    inputs = metile.tensor(Input, shape=(SIZE,), access="read")
    ups = metile.tensor(Up, shape=(SIZE,), access="read")
    outputs = metile.tensor(Output, shape=(SIZE,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    value = metile.cast(inputs.load((positions,)), "f32")
    activated, _ = _activation(value, KIND)
    if GATED:
        activated = activated * metile.cast(ups.load((positions,)), "f32")
    outputs.store((positions,), activated)


@metile.kernel
def activation_backward(
    Input,
    Up,
    GradOutput,
    GradInput,
    GradUp,
    SIZE,
    *,
    KIND: metile.constexpr = "silu",
    GATED: metile.constexpr = False,
    BLOCK: metile.constexpr = 128,
):
    _validate(KIND, GATED, BLOCK)
    inputs = metile.tensor(Input, shape=(SIZE,), access="read")
    ups = metile.tensor(Up, shape=(SIZE,), access="read")
    output_gradients = metile.tensor(GradOutput, shape=(SIZE,), access="read")
    input_gradients = metile.tensor(GradInput, shape=(SIZE,), access="write")
    up_gradients = metile.tensor(GradUp, shape=(SIZE,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    value = metile.cast(inputs.load((positions,)), "f32")
    gradient = metile.cast(output_gradients.load((positions,)), "f32")
    activated, derivative = _activation(value, KIND)
    if GATED:
        up = metile.cast(ups.load((positions,)), "f32")
        input_gradients.store((positions,), gradient * up * derivative)
        up_gradients.store((positions,), gradient * activated)
    else:
        input_gradients.store((positions,), gradient * derivative)
