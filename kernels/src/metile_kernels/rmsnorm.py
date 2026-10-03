import metile


@metile.kernel
def rmsnorm_register(
    X, W, Out, eps, N: metile.constexpr, BLOCK: metile.constexpr, LAYOUT: metile.constexpr = None
):
    if type(N) is not int or not 0 < N <= BLOCK:
        raise ValueError("register RMSNorm requires 0 < N <= BLOCK")
    row = metile.program_id(0)
    inputs = metile.tensor(X + row * N, shape=(N,), access="read")
    weights = metile.tensor(W, shape=(N,), access="read")
    outputs = metile.tensor(Out + row * N, shape=(N,), access="write")
    layout = LAYOUT or metile.ThreadLayout.identity(BLOCK, elements_per_thread=4)
    columns = metile.arange(0, BLOCK, layout=layout)
    values = metile.cast(inputs.load((columns,)), "f32")
    total = metile.sum(values * values)
    reciprocal = 1.0 / metile.sqrt(total / N + eps)
    scale = metile.cast(weights.load((columns,)), "f32")
    outputs.store((columns,), values * reciprocal * scale)


@metile.kernel
def rmsnorm(X, W, Out, N, eps, BLOCK: metile.constexpr):
    row = metile.program_id(0)
    inputs = metile.tensor(X + row * N, shape=(N,), access="read")
    weights = metile.tensor(W, shape=(N,), access="read")
    outputs = metile.tensor(Out + row * N, shape=(N,), access="write")

    ss = 0.0
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        x = inputs.load((cols,))
        x_f32 = metile.cast(x, "f32")
        ss = ss + x_f32 * x_f32

    ss = metile.sum(ss)

    rms = 1.0 / metile.sqrt(ss / N + eps)

    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        x = inputs.load((cols,))
        w = weights.load((cols,))
        result = metile.cast(x, "f32") * rms * metile.cast(w, "f32")
        outputs.store((cols,), result)
