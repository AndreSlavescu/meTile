import metile


@metile.kernel
def add_rmsnorm(X, Residual, W, Sum, Out, N, eps, BLOCK: metile.constexpr):
    row = metile.program_id(0)

    sum_squares = 0.0
    for start in metile.tile_range(0, N, BLOCK):
        columns = start + metile.arange(0, BLOCK)
        mask = columns < N
        values = metile.cast(metile.load(X + row * N + columns, mask=mask), "f32")
        residual = metile.cast(metile.load(Residual + row * N + columns, mask=mask), "f32")
        summed = values + residual
        metile.store(Sum + row * N + columns, summed, mask=mask)
        sum_squares = sum_squares + summed * summed

    inverse_rms = 1.0 / metile.sqrt(metile.sum(sum_squares) / N + eps)

    for start in metile.tile_range(0, N, BLOCK):
        columns = start + metile.arange(0, BLOCK)
        mask = columns < N
        summed = metile.cast(metile.load(Sum + row * N + columns, mask=mask), "f32")
        weight = metile.cast(metile.load(W + columns, mask=mask), "f32")
        metile.store(Out + row * N + columns, summed * inverse_rms * weight, mask=mask)
