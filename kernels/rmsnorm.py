import metile


@metile.kernel
def rmsnorm(X, W, Out, N, eps, BLOCK: metile.constexpr):
    row = metile.program_id(0)

    ss = 0.0
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        x_f32 = metile.cast(x, "f32")
        ss = ss + x_f32 * x_f32

    ss = metile.sum(ss)

    rms = 1.0 / metile.sqrt(ss / N + eps)

    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        w = metile.load(W + cols, mask=mask)
        result = metile.cast(x, "f32") * rms * metile.cast(w, "f32")
        metile.store(Out + row * N + cols, result, mask=mask)
