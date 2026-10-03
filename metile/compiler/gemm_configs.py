"""Measured GEMM search policy, separate from tensor program definitions."""


def _candidate(block_m, block_n, *, block_k=16, swizzle="auto", **options):
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "WM": block_m // 32,
        "WN": block_n // 32,
        "SWIZZLE": swizzle,
        **options,
    }


def matmul_candidates():
    """Return the existing ordered M5 candidate family as compiler options."""
    candidates = [
        _candidate(64, 64, swizzle=schedule)
        for schedule in (
            "linear",
            "diagonal",
            "morton",
            "hilbert",
            "grouped2",
            "grouped4",
            "grouped8",
        )
    ]
    candidates.extend(
        _candidate(64, 64, swizzle="grouped8", NAX_FRAGMENTS=True, **options)
        for options in ({}, {"NAX_OUTER_K": 512})
    )
    candidates.append(_candidate(128, 128, swizzle="grouped4", NAX_FRAGMENTS=True, NAX_OUTER_K=128))
    for block_m, block_n, schedule, outer_k in (
        (128, 128, "grouped4", 256),
        (64, 128, "morton", 256),
        (64, 128, "morton", 512),
        (128, 64, "morton", 512),
        (128, 64, "hilbert", 512),
        (256, 64, "morton", 512),
    ):
        candidates.append(
            _candidate(
                block_m,
                block_n,
                swizzle=schedule,
                NAX_FRAGMENTS=True,
                NAX_OUTER_K=outer_k,
                NAX_K_UNROLL=2,
            )
        )
    for schedule, outer_k, options in (
        ("grouped4", 512, {"NAX_SKIP_FIRST_EPOCH_BARRIER": True}),
        ("grouped4", 512, {"NAX_TRAILING_EPOCH_BARRIER": True}),
        ("hilbert", 512, {"NAX_TRAILING_EPOCH_BARRIER": True}),
        ("grouped4", 512, {}),
        ("hilbert", 512, {}),
        ("grouped4", 1024, {}),
        ("grouped4", 1024, {"NAX_TRAILING_EPOCH_BARRIER": True}),
    ):
        candidates.append(
            _candidate(
                128,
                128,
                swizzle=schedule,
                NAX_FRAGMENTS=True,
                NAX_OUTER_K=outer_k,
                NAX_K_UNROLL=2,
                **options,
            )
        )
    candidates.extend(
        (
            _candidate(64, 64, block_k=32),
            _candidate(64, 128),
            _candidate(64, 128, swizzle="grouped8"),
            _candidate(128, 64, block_k=32),
            _candidate(128, 128, block_k=32),
        )
    )
    return candidates
