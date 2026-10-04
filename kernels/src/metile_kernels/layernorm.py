import metile


@metile.kernel
def layernorm(X, W, B, Out, N, BLOCK: metile.constexpr):
    row = metile.program_id(0)
    inputs = metile.tensor(X + row * N, shape=(N,), access="read")
    weights = metile.tensor(W, shape=(N,), access="read")
    biases = metile.tensor(B, shape=(N,), access="read")
    outputs = metile.tensor(Out + row * N, shape=(N,), access="write")

    s = 0.0
    ss = 0.0
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        x = inputs.load((cols,))
        s = s + x
        ss = ss + x * x
    mean = metile.sum(s) / N
    inv_std = 1.0 / metile.sqrt(metile.sum(ss) / N - mean * mean + 1e-5)

    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        x = inputs.load((cols,))
        w = weights.load((cols,))
        b = biases.load((cols,))
        outputs.store((cols,), (x - mean) * inv_std * w + b)
