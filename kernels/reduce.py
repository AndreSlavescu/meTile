import metile


@metile.kernel
def reduce_2(p0, p1, out, N_ELEM, BLOCK_SIZE: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK_SIZE + metile.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM
    acc = metile.load(p0 + offs, mask=mask) + metile.load(p1 + offs, mask=mask)
    metile.store(out + offs, acc, mask=mask)


@metile.kernel
def reduce_4(p0, p1, p2, p3, out, N_ELEM, BLOCK_SIZE: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK_SIZE + metile.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM
    acc = metile.load(p0 + offs, mask=mask) + metile.load(p1 + offs, mask=mask)
    acc = acc + metile.load(p2 + offs, mask=mask) + metile.load(p3 + offs, mask=mask)
    metile.store(out + offs, acc, mask=mask)


@metile.kernel
def reduce_8(p0, p1, p2, p3, p4, p5, p6, p7, out, N_ELEM, BLOCK_SIZE: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK_SIZE + metile.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM
    acc = metile.load(p0 + offs, mask=mask) + metile.load(p1 + offs, mask=mask)
    acc = acc + metile.load(p2 + offs, mask=mask) + metile.load(p3 + offs, mask=mask)
    acc = acc + metile.load(p4 + offs, mask=mask) + metile.load(p5 + offs, mask=mask)
    acc = acc + metile.load(p6 + offs, mask=mask) + metile.load(p7 + offs, mask=mask)
    metile.store(out + offs, acc, mask=mask)


@metile.kernel
def reduce_16(
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
    out, N_ELEM, BLOCK_SIZE: metile.constexpr,
):
    pid = metile.program_id(0)
    offs = pid * BLOCK_SIZE + metile.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM
    acc = metile.load(p0 + offs, mask=mask) + metile.load(p1 + offs, mask=mask)
    acc = acc + metile.load(p2 + offs, mask=mask) + metile.load(p3 + offs, mask=mask)
    acc = acc + metile.load(p4 + offs, mask=mask) + metile.load(p5 + offs, mask=mask)
    acc = acc + metile.load(p6 + offs, mask=mask) + metile.load(p7 + offs, mask=mask)
    acc = acc + metile.load(p8 + offs, mask=mask) + metile.load(p9 + offs, mask=mask)
    acc = acc + metile.load(p10 + offs, mask=mask) + metile.load(p11 + offs, mask=mask)
    acc = acc + metile.load(p12 + offs, mask=mask) + metile.load(p13 + offs, mask=mask)
    acc = acc + metile.load(p14 + offs, mask=mask) + metile.load(p15 + offs, mask=mask)
    metile.store(out + offs, acc, mask=mask)


REDUCE_KERNELS = {2: reduce_2, 4: reduce_4, 8: reduce_8, 16: reduce_16}
