"""Functional FP32 cotangent accumulation for native attention graph VJPs."""

import metile


@metile.kernel
def attention_graph_gradient_add(
    Left,
    Right,
    Out,
    ELEMENTS,
    *,
    ADD_RIGHT: metile.constexpr = True,
    BLOCK: metile.constexpr = 256,
):
    if type(ADD_RIGHT) is not bool or type(BLOCK) is not int or BLOCK != 256:
        raise ValueError("gradient accumulation requires bool ADD_RIGHT and BLOCK=256")
    left = metile.tensor(Left, shape=(ELEMENTS,), access="read")
    right = metile.tensor(Right, shape=(ELEMENTS,), access="read")
    output = metile.tensor(Out, shape=(ELEMENTS,), access="write")
    indices = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    result = metile.cast(left.load((indices,)), "f32")
    if ADD_RIGHT:
        result = result + metile.cast(right.load((indices,)), "f32")
    output.store((indices,), result)
