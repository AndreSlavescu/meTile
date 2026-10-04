"""Strict FP32 normalization and cross-entropy primitives with explicit VJPs.

Cross entropy uses uniform label smoothing and adds z_loss * logsumexp(x)**2
on non-ignored rows only. Mean reduction divides by the number of valid rows.
The implementation uses fixed-order partial sums and never floating atomics.
"""

import metile


def _row_log_normalizer(inputs, row, columns, block):
    lanes = metile.arange(0, block)
    maximum = metile.loop_state(float("-inf"))
    for start in metile.tile_range(0, columns, block):
        values = inputs.load((row, start + lanes), other=float("-inf"))
        maximum.update(metile.maximum(maximum.value, metile.max(values)))
    total = metile.loop_state(0.0)
    for start in metile.tile_range(0, columns, block):
        values = inputs.load((row, start + lanes), other=float("-inf"))
        total.update(total.value + metile.sum(metile.exp(values - maximum.value)))
    return maximum.value, metile.log(total.value)


@metile.kernel
def row_normalization_forward_kernel(
    source,
    destination,
    rows,
    columns,
    LOGARITHMIC: metile.constexpr,
    BLOCK: metile.constexpr,
):
    inputs = metile.tensor(source, shape=(rows, columns), access="read")
    outputs = metile.tensor(destination, shape=(rows, columns), access="write")
    row = metile.program_id(0)
    maximum, log_total = _row_log_normalizer(inputs, row, columns, BLOCK)
    lanes = metile.arange(0, BLOCK)
    for start in metile.tile_range(0, columns, BLOCK):
        positions = start + lanes
        centered = inputs.load((row, positions)) - maximum
        if LOGARITHMIC:
            result = centered - log_total
        else:
            result = metile.exp(centered - log_total)
        outputs.store((row, positions), result)


@metile.kernel
def row_normalization_backward_kernel(
    output,
    output_gradient,
    input_gradient,
    rows,
    columns,
    LOGARITHMIC: metile.constexpr,
    BLOCK: metile.constexpr,
):
    outputs = metile.tensor(output, shape=(rows, columns), access="read")
    seeds = metile.tensor(output_gradient, shape=(rows, columns), access="read")
    gradients = metile.tensor(input_gradient, shape=(rows, columns), access="write")
    row = metile.program_id(0)
    lanes = metile.arange(0, BLOCK)
    total = metile.loop_state(0.0)
    for start in metile.tile_range(0, columns, BLOCK):
        positions = start + lanes
        seed = seeds.load((row, positions))
        if LOGARITHMIC:
            contribution = seed
        else:
            contribution = seed * outputs.load((row, positions))
        total.update(total.value + metile.sum(contribution))
    for start in metile.tile_range(0, columns, BLOCK):
        positions = start + lanes
        seed = seeds.load((row, positions))
        saved = outputs.load((row, positions))
        if LOGARITHMIC:
            gradient = seed - metile.exp(saved) * total.value
        else:
            gradient = saved * (seed - total.value)
        gradients.store((row, positions), gradient)


@metile.kernel
def cross_entropy_forward_kernel(
    source,
    target,
    row_loss,
    log_normalizer,
    maxima,
    log_totals,
    rows,
    columns,
    ignore_index,
    smoothing,
    z_loss,
    SMOOTH: metile.constexpr,
    BLOCK: metile.constexpr,
):
    inputs = metile.tensor(source, shape=(rows, columns), access="read")
    targets = metile.tensor(target, shape=(rows,), access="read")
    losses = metile.tensor(row_loss, shape=(rows, 1), access="write")
    normalizers = metile.tensor(log_normalizer, shape=(rows, 1), access="write")
    saved_maxima = metile.tensor(maxima, shape=(rows, 1), access="write")
    saved_log_totals = metile.tensor(log_totals, shape=(rows, 1), access="write")
    row = metile.program_id(0)
    maximum, log_total = _row_log_normalizer(inputs, row, columns, BLOCK)
    log_sum = maximum + log_total
    label = targets.load(row)
    target_value = inputs.load((row, label))
    loss = (maximum - target_value) + log_total
    lanes = metile.arange(0, BLOCK)
    if SMOOTH:
        centered_total = metile.loop_state(0.0)
        for start in metile.tile_range(0, columns, BLOCK):
            positions = start + lanes
            centered = inputs.load((row, positions)) - maximum
            centered = metile.where(positions < columns, centered, 0.0)
            centered_total.update(centered_total.value + metile.sum(centered))
        uniform_loss = log_total - centered_total.value / columns
        loss = (1.0 - smoothing) * loss + smoothing * uniform_loss
    loss = loss + z_loss * log_sum * log_sum
    losses.store((row, lanes), metile.where(label != ignore_index, loss, 0.0))
    normalizers.store((row, lanes), log_sum)
    saved_maxima.store((row, lanes), maximum)
    saved_log_totals.store((row, lanes), log_total)


@metile.kernel
def cross_entropy_reduce_kernel(row_loss, loss, rows, normalizer, BLOCK: metile.constexpr):
    inputs = metile.tensor(row_loss, shape=(rows,), access="read")
    output = metile.tensor(loss, shape=(1,), access="write")
    lanes = metile.arange(0, BLOCK)
    total = metile.loop_state(0.0)
    for start in metile.tile_range(0, rows, BLOCK):
        total.update(total.value + metile.sum(inputs.load(start + lanes)))
    output.store(lanes, total.value / normalizer)


@metile.kernel
def cross_entropy_backward_kernel(
    source,
    target,
    maxima,
    log_totals,
    cotangent,
    input_gradient,
    rows,
    columns,
    ignore_index,
    smoothing,
    z_loss,
    normalizer,
    REDUCED: metile.constexpr,
    BLOCK: metile.constexpr,
):
    inputs = metile.tensor(source, shape=(rows, columns), access="read")
    targets = metile.tensor(target, shape=(rows,), access="read")
    saved_maxima = metile.tensor(maxima, shape=(rows,), access="read")
    saved_log_totals = metile.tensor(log_totals, shape=(rows,), access="read")
    seeds = metile.tensor(cotangent, shape=(1 if REDUCED else rows,), access="read")
    gradients = metile.tensor(input_gradient, shape=(rows, columns), access="write")
    row = metile.program_id(0)
    label = targets.load(row)
    maximum = saved_maxima.load(row)
    log_total = saved_log_totals.load(row)
    factor = seeds.load(0 if REDUCED else row) / normalizer
    coefficient = 1.0 + 2.0 * z_loss * (maximum + log_total)
    lanes = metile.arange(0, BLOCK)
    for start in metile.tile_range(0, columns, BLOCK):
        positions = start + lanes
        probability = metile.exp((inputs.load((row, positions)) - maximum) - log_total)
        target_probability = smoothing / columns + metile.where(
            positions == label, 1.0 - smoothing, 0.0
        )
        gradient = factor * (coefficient * probability - target_probability)
        gradients.store((row, positions), metile.where(label != ignore_index, gradient, 0.0))
