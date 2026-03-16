import metile


@metile.kernel
def exp_sqrt_kernel(X, out_exp, out_sqrt, N, BLOCK: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK + metile.arange(0, BLOCK)
    mask = offs < N

    x = metile.load(X + offs, mask=mask)

    with metile.simdgroup_role(role=0, num_roles=2):
        metile.store(out_exp + offs, metile.exp(x), mask=mask)

    with metile.simdgroup_role(role=1, num_roles=2):
        metile.store(out_sqrt + offs, metile.sqrt(metile.abs(x)), mask=mask)


@metile.kernel
def gelu_silu_kernel(X, out_gelu, out_silu, N, BLOCK: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK + metile.arange(0, BLOCK)
    mask = offs < N

    x = metile.load(X + offs, mask=mask)

    with metile.simdgroup_role(role=0, num_roles=2):
        gelu = x / (1.0 + metile.exp(0.0 - 1.702 * x))
        metile.store(out_gelu + offs, gelu, mask=mask)

    with metile.simdgroup_role(role=1, num_roles=2):
        silu = x / (1.0 + metile.exp(0.0 - x))
        metile.store(out_silu + offs, silu, mask=mask)


@metile.kernel
def geglu_specialized_kernel(X_gate, X_up, out, N, BLOCK: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK + metile.arange(0, BLOCK)
    mask = offs < N

    gate = metile.load(X_gate + offs, mask=mask)
    up = metile.load(X_up + offs, mask=mask)

    with metile.simdgroup_role(role=0, num_roles=2):
        gelu_gate = gate / (1.0 + metile.exp(0.0 - 1.702 * gate))
        metile.store(out + offs, gelu_gate * up, mask=mask)

    with metile.simdgroup_role(role=1, num_roles=2):
        gelu_gate = gate / (1.0 + metile.exp(0.0 - 1.702 * gate))
        metile.store(out + offs, gelu_gate * up, mask=mask)


@metile.kernel
def exp_kernel(X, out, N, BLOCK: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK + metile.arange(0, BLOCK)
    mask = offs < N
    metile.store(out + offs, metile.exp(metile.load(X + offs, mask=mask)), mask=mask)


@metile.kernel
def sqrt_abs_kernel(X, out, N, BLOCK: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK + metile.arange(0, BLOCK)
    mask = offs < N
    metile.store(out + offs, metile.sqrt(metile.abs(metile.load(X + offs, mask=mask))), mask=mask)


@metile.kernel
def gelu_kernel(X, out, N, BLOCK: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK + metile.arange(0, BLOCK)
    mask = offs < N
    x = metile.load(X + offs, mask=mask)
    metile.store(out + offs, x / (1.0 + metile.exp(0.0 - 1.702 * x)), mask=mask)


@metile.kernel
def silu_kernel(X, out, N, BLOCK: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK + metile.arange(0, BLOCK)
    mask = offs < N
    x = metile.load(X + offs, mask=mask)
    metile.store(out + offs, x / (1.0 + metile.exp(0.0 - x)), mask=mask)


@metile.kernel
def geglu_kernel(X_gate, X_up, out, N, BLOCK: metile.constexpr):
    pid = metile.program_id(0)
    offs = pid * BLOCK + metile.arange(0, BLOCK)
    mask = offs < N
    gate = metile.load(X_gate + offs, mask=mask)
    up = metile.load(X_up + offs, mask=mask)
    gelu_gate = gate / (1.0 + metile.exp(0.0 - 1.702 * gate))
    metile.store(out + offs, gelu_gate * up, mask=mask)
