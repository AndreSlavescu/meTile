"""Backend-agnostic machinery for choosing between kernels by measurement.

None of this knows what a kernel is or how to run one. Callers supply a `measure` function
that turns a thunk into seconds, and everything here is about the shape of the comparison:
what order to time things in, how many candidates to time together, and when a measured
difference is large enough to act on.

It lives outside metile.backends because it is not specific to any one of them. The MLX
backend is one consumer; the native Metal runtime in metile.runtime is another.
"""

from metile.tuning.tournament import (
    confirm_pairwise,
    pessimistic,
    round_robin,
    select_fastest,
    token_bucket,
)

__all__ = [
    "confirm_pairwise",
    "pessimistic",
    "round_robin",
    "select_fastest",
    "token_bucket",
]
