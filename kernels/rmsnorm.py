import metile


@metile.kernel
def rmsnorm(X, W, Out, N, BLOCK: metile.constexpr):
    row = metile.program_id(0)

    ss = 0.0
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        ss = ss + x * x

    ss = metile.sum(ss)

    rms = 1.0 / metile.sqrt(ss / N + 1e-5)

    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        w = metile.load(W + cols, mask=mask)
        metile.store(Out + row * N + cols, x * rms * w, mask=mask)
