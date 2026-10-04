"""Full/partial rotary embeddings and adjoints for split-half/interleaved pairs.

Input/output are [ROWS,DIM]. Cosine and sine inputs are already gathered and
expanded to [ROWS,ROTARY_DIM/2]; pairing is controlled by INTERLEAVED. This
kernel does not select position IDs, construct frequencies, or assume unit
cosine/sine norm. Backward is the transpose of the actual supplied transform,
not its inverse, and returns gradients for inputs AND both coefficient arrays.
Broadcasted tables require an additional caller-owned reduction of coefficient
gradients. Unrotated dimensions are copied in both directions.

Launch ROWS groups with BLOCK>=max(ROTARY_DIM/2,DIM-ROTARY_DIM), a multiple
of 32 no larger than1024. Use disjoint FP16/FP32 storage, FP32 gradient outputs,
and STRICT_MATH=True. Mathematical reference: https://arxiv.org/abs/2104.09864.
"""

import metile


def _contract(dimension, rotary, interleaved, block):
    if type(dimension) is not int or dimension <= 0 or dimension > 1024:
        raise ValueError("DIM must be a positive integer <=1024")
    if type(rotary) is not int or rotary <= 0 or rotary > dimension or rotary % 2:
        raise ValueError("ROTARY_DIM must be positive, even and <=DIM")
    if type(interleaved) is not bool:
        raise TypeError("INTERLEAVED must be bool")
    if (
        type(block) is not int
        or block % 32
        or not max(rotary // 2, dimension - rotary, 32) <= block <= 1024
    ):
        raise ValueError("BLOCK must cover pairs and the unrotated tail, in multiples of32")


def _pair_views(pointer, rows, dimension, rotary, interleaved, access):
    stride = 2 if interleaved else 1
    second = 1 if interleaved else rotary // 2
    first_view = metile.tensor(
        pointer, shape=(rows, rotary // 2), strides=(dimension, stride), access=access
    )
    second_view = metile.tensor(
        pointer + second, shape=(rows, rotary // 2), strides=(dimension, stride), access=access
    )
    tail_view = metile.tensor(
        pointer + rotary, shape=(rows, dimension - rotary), strides=(dimension, 1), access=access
    )
    return first_view, second_view, tail_view


@metile.kernel
def rope_forward(
    Input,
    Cosine,
    Sine,
    Output,
    ROWS,
    *,
    DIM: metile.constexpr,
    ROTARY_DIM: metile.constexpr,
    INTERLEAVED: metile.constexpr = False,
    BLOCK: metile.constexpr = 128,
):
    _contract(DIM, ROTARY_DIM, INTERLEAVED, BLOCK)
    left, right, tail = _pair_views(Input, ROWS, DIM, ROTARY_DIM, INTERLEAVED, "read")
    out_left, out_right, out_tail = _pair_views(Output, ROWS, DIM, ROTARY_DIM, INTERLEAVED, "write")
    cosines = metile.tensor(Cosine, shape=(ROWS, ROTARY_DIM // 2), access="read")
    sines = metile.tensor(Sine, shape=(ROWS, ROTARY_DIM // 2), access="read")
    row = metile.program_id(0)
    positions = metile.arange(0, BLOCK)
    first = metile.cast(left.load((row, positions)), "f32")
    second = metile.cast(right.load((row, positions)), "f32")
    cosine = metile.cast(cosines.load((row, positions)), "f32")
    sine = metile.cast(sines.load((row, positions)), "f32")
    out_left.store((row, positions), first * cosine - second * sine)
    out_right.store((row, positions), second * cosine + first * sine)
    out_tail.store((row, positions), tail.load((row, positions)))


@metile.kernel
def rope_backward(
    Input,
    Cosine,
    Sine,
    GradOutput,
    GradInput,
    GradCosine,
    GradSine,
    ROWS,
    *,
    DIM: metile.constexpr,
    ROTARY_DIM: metile.constexpr,
    INTERLEAVED: metile.constexpr = False,
    BLOCK: metile.constexpr = 128,
):
    _contract(DIM, ROTARY_DIM, INTERLEAVED, BLOCK)
    left, right, _ = _pair_views(Input, ROWS, DIM, ROTARY_DIM, INTERLEAVED, "read")
    grad_left, grad_right, grad_tail = _pair_views(
        GradOutput, ROWS, DIM, ROTARY_DIM, INTERLEAVED, "read"
    )
    out_left, out_right, out_tail = _pair_views(
        GradInput, ROWS, DIM, ROTARY_DIM, INTERLEAVED, "write"
    )
    cosines = metile.tensor(Cosine, shape=(ROWS, ROTARY_DIM // 2), access="read")
    sines = metile.tensor(Sine, shape=(ROWS, ROTARY_DIM // 2), access="read")
    grad_cosines = metile.tensor(GradCosine, shape=(ROWS, ROTARY_DIM // 2), access="write")
    grad_sines = metile.tensor(GradSine, shape=(ROWS, ROTARY_DIM // 2), access="write")
    row = metile.program_id(0)
    positions = metile.arange(0, BLOCK)
    first = metile.cast(left.load((row, positions)), "f32")
    second = metile.cast(right.load((row, positions)), "f32")
    grad_first = metile.cast(grad_left.load((row, positions)), "f32")
    grad_second = metile.cast(grad_right.load((row, positions)), "f32")
    cosine = metile.cast(cosines.load((row, positions)), "f32")
    sine = metile.cast(sines.load((row, positions)), "f32")
    out_left.store((row, positions), grad_first * cosine + grad_second * sine)
    out_right.store((row, positions), grad_second * cosine - grad_first * sine)
    out_tail.store((row, positions), grad_tail.load((row, positions)))
    grad_cosines.store((row, positions), grad_first * first + grad_second * second)
    grad_sines.store((row, positions), grad_second * first - grad_first * second)
