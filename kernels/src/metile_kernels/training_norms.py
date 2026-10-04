"""FP32-accumulating normalization kernels with deterministic parameter adjoints."""

import metile


@metile.kernel
def norm_forward_kernel(
    source,
    residual,
    weight,
    bias,
    output,
    residual_output,
    means,
    inverse_scales,
    rows,
    columns,
    epsilon,
    KIND: metile.constexpr,
    BLOCK: metile.constexpr,
    LAYOUT: metile.constexpr,
):
    if KIND not in ("rms", "layer", "add_rms"):
        raise ValueError("unknown training normalization kind")
    inputs = metile.tensor(source, shape=(rows, columns), access="read")
    residuals = metile.tensor(residual, shape=(rows, columns), access="read")
    weights = metile.tensor(weight, shape=(columns,), access="read")
    biases = metile.tensor(bias, shape=(columns,), access="read")
    outputs = metile.tensor(output, shape=(rows, columns), access="write")
    sums = metile.tensor(residual_output, shape=(rows, columns), access="write")
    saved_means = metile.tensor(means, shape=(rows, 1), access="write")
    saved_scales = metile.tensor(inverse_scales, shape=(rows, 1), access="write")
    row = metile.program_id(0)
    positions = metile.arange(0, BLOCK, layout=LAYOUT)
    values = metile.cast(inputs.load((row, positions)), "f32")
    if KIND == "add_rms":
        values = values + metile.cast(residuals.load((row, positions)), "f32")
        sums.store((row, positions), values)
    mean = metile.sum(values) / columns if KIND == "layer" else metile.scalar(0.0)
    centered = metile.where(positions < columns, values - mean, 0.0)
    inverse = 1.0 / metile.sqrt(metile.sum(centered * centered) / columns + epsilon)
    result = centered * inverse * metile.cast(weights.load((positions,)), "f32")
    if KIND == "layer":
        result = result + metile.cast(biases.load((positions,)), "f32")
    outputs.store((row, positions), result)
    saved_means.store((row, positions), mean)
    saved_scales.store((row, positions), inverse)


@metile.kernel
def norm_backward_rows_kernel(
    source,
    residual,
    weight,
    means,
    inverse_scales,
    output_gradient,
    residual_output_gradient,
    source_gradient,
    residual_gradient,
    weight_partials,
    bias_partials,
    rows,
    columns,
    KIND: metile.constexpr,
    BLOCK: metile.constexpr,
    LAYOUT: metile.constexpr,
):
    if KIND not in ("rms", "layer", "add_rms"):
        raise ValueError("unknown training normalization kind")
    inputs = metile.tensor(source, shape=(rows, columns), access="read")
    residuals = metile.tensor(residual, shape=(rows, columns), access="read")
    weights = metile.tensor(weight, shape=(columns,), access="read")
    saved_means = metile.tensor(means, shape=(rows, 1), access="read")
    saved_scales = metile.tensor(inverse_scales, shape=(rows, 1), access="read")
    seeds = metile.tensor(output_gradient, shape=(rows, columns), access="read")
    residual_seeds = metile.tensor(residual_output_gradient, shape=(rows, columns), access="read")
    source_gradients = metile.tensor(source_gradient, shape=(rows, columns), access="write")
    residual_gradients = metile.tensor(residual_gradient, shape=(rows, columns), access="write")
    weight_rows = metile.tensor(weight_partials, shape=(rows, columns), access="write")
    bias_rows = metile.tensor(bias_partials, shape=(rows, columns), access="write")
    row = metile.program_id(0)
    positions = metile.arange(0, BLOCK, layout=LAYOUT)
    values = metile.cast(inputs.load((row, positions)), "f32")
    if KIND == "add_rms":
        values = values + metile.cast(residuals.load((row, positions)), "f32")
    mean = saved_means.load((row, 0))
    inverse = saved_scales.load((row, 0))
    centered = metile.where(positions < columns, values - mean, 0.0)
    seed = metile.cast(seeds.load((row, positions)), "f32")
    weighted = seed * metile.cast(weights.load((positions,)), "f32")
    projection = metile.sum(weighted * centered) / columns
    input_gradient = weighted - centered * inverse * inverse * projection
    if KIND == "layer":
        input_gradient = input_gradient - metile.sum(weighted) / columns
        bias_rows.store((row, positions), seed)
    input_gradient = input_gradient * inverse
    if KIND == "add_rms":
        input_gradient = input_gradient + metile.cast(residual_seeds.load((row, positions)), "f32")
        residual_gradients.store((row, positions), input_gradient)
    source_gradients.store((row, positions), input_gradient)
    weight_rows.store((row, positions), seed * centered * inverse)


@metile.kernel
def norm_parameter_reduce_kernel(
    weight_partials,
    bias_partials,
    weight_gradient,
    bias_gradient,
    rows,
    columns,
    HAS_BIAS: metile.constexpr,
    BLOCK: metile.constexpr,
):
    weight_rows = metile.tensor(weight_partials, shape=(rows, columns), access="read")
    bias_rows = metile.tensor(bias_partials, shape=(rows, columns), access="read")
    weight_gradients = metile.tensor(weight_gradient, shape=(columns,), access="write")
    bias_gradients = metile.tensor(bias_gradient, shape=(columns,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    zero = metile.where(positions < columns, 0.0, 0.0)
    weight_total = metile.loop_state(zero)
    if HAS_BIAS:
        bias_total = metile.loop_state(zero)
    for row in metile.tile_range(0, rows, 1):
        weight_total.update(weight_total.value + weight_rows.load((row, positions)))
        if HAS_BIAS:
            bias_total.update(bias_total.value + bias_rows.load((row, positions)))
    weight_gradients.store((positions,), weight_total.value)
    if HAS_BIAS:
        bias_gradients.store((positions,), bias_total.value)
