"""Bounds-checked copy/cast/transpose used to compose GEMM adjoints."""

import metile


@metile.kernel
def matrix_pack(
    Input,
    Output,
    ROWS,
    COLUMNS,
    *,
    TRANSPOSE: metile.constexpr = False,
    BLOCK: metile.constexpr = 128,
):
    if type(TRANSPOSE) is not bool:
        raise TypeError("TRANSPOSE must be bool")
    if type(BLOCK) is not int or BLOCK < 32 or BLOCK > 1024 or BLOCK % 32:
        raise ValueError("BLOCK must be a multiple of32 in[32,1024]")
    source = metile.tensor(Input, shape=(ROWS, COLUMNS), access="read")
    destination = metile.tensor(
        Output, shape=(COLUMNS, ROWS) if TRANSPOSE else (ROWS, COLUMNS), access="write"
    )
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    row = positions // COLUMNS
    column = positions % COLUMNS
    value = source.load((row, column))
    if TRANSPOSE:
        destination.store((column, row), value)
    else:
        destination.store((row, column), value)
