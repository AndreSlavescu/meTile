import metile


@metile.kernel
def layernorm(X, W, B, Out, N, BLOCK: metile.constexpr):
    row = metile.program_id(0)

    s = 0.0
    ss = 0.0
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        s = s + x
        ss = ss + x * x
    mean = metile.sum(s) / N
    inv_std = 1.0 / metile.sqrt(metile.sum(ss) / N - mean * mean + 1e-5)

    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        w = metile.load(W + cols, mask=mask)
        b = metile.load(B + cols, mask=mask)
        metile.store(Out + row * N + cols, (x - mean) * inv_std * w + b, mask=mask)
