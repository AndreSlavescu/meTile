from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise

from metile.compiler.schedules import SCHEDULES, resolve_schedule, schedule_coordinates
from metile.ir import metal_ir as mir


@dataclass(frozen=True)
class FinitePermutation:
    """An element of the symmetric group over a finite dispatch grid."""

    images: tuple[int, ...]

    def __post_init__(self):
        if sorted(self.images) != list(range(len(self.images))):
            raise ValueError("permutation must contain each index exactly once")

    @classmethod
    def identity(cls, size: int) -> FinitePermutation:
        return cls(tuple(range(size)))

    def compose(self, other: FinitePermutation) -> FinitePermutation:
        if len(self.images) != len(other.images):
            raise ValueError("permutations must act on the same set")
        return FinitePermutation(tuple(self.images[index] for index in other.images))

    def inverse(self) -> FinitePermutation:
        inverse = [0] * len(self.images)
        for source, destination in enumerate(self.images):
            inverse[destination] = source
        return FinitePermutation(tuple(inverse))

    @cached_property
    def cycles(self) -> tuple[tuple[int, ...], ...]:
        remaining = set(range(len(self.images)))
        cycles = []
        while remaining:
            start = min(remaining)
            cycle = []
            current = start
            while current in remaining:
                remaining.remove(current)
                cycle.append(current)
                current = self.images[current]
            cycles.append(tuple(cycle))
        return tuple(cycles)


@dataclass(frozen=True)
class GridSymmetryGroup:
    """The layout-preserving reflection group acting on a rectangular grid."""

    grid_m: int
    grid_n: int

    @cached_property
    def elements(self) -> tuple[FinitePermutation, ...]:
        transformations = (
            lambda m, n: (m, n),
            lambda m, n: (self.grid_m - 1 - m, n),
            lambda m, n: (m, self.grid_n - 1 - n),
            lambda m, n: (self.grid_m - 1 - m, self.grid_n - 1 - n),
        )
        elements = []
        for transform in transformations:
            images = []
            for index in range(self.grid_m * self.grid_n):
                m, n = divmod(index, self.grid_n)
                target_m, target_n = transform(m, n)
                images.append(target_m * self.grid_n + target_n)
            elements.append(FinitePermutation(tuple(images)))
        return tuple(dict.fromkeys(elements))

    @cached_property
    def fundamental_domain(self) -> tuple[int, ...]:
        """Return one canonical representative from every group orbit."""
        unassigned = set(range(self.grid_m * self.grid_n))
        representatives = []
        while unassigned:
            seed = min(unassigned)
            orbit = {element.images[seed] for element in self.elements}
            representatives.append(min(orbit))
            unassigned.difference_update(orbit)
        return tuple(representatives)

    def canonicalize(self, permutation: FinitePermutation) -> tuple[int, ...]:
        """Canonicalize a traversal up to reflection-group conjugacy."""
        conjugates = []
        for symmetry in self.elements:
            conjugate = symmetry.compose(permutation).compose(symmetry.inverse())
            conjugates.append(conjugate.images)
        return min(conjugates)


@dataclass(frozen=True)
class ScheduleCost:
    pattern: str
    description_bits: int
    locality_cost: float
    worst_jump: int
    score: float
    permutation: FinitePermutation
    canonical_signature: tuple[int, ...]


_DESCRIPTION_BITS = {
    "linear": 8,
    "diagonal": 28,
    "morton": 68,
    "hilbert": 92,
}


def analyze_schedule(pattern: str, grid_m: int, grid_n: int) -> ScheduleCost:
    coordinates = list(schedule_coordinates(grid_m, grid_n, pattern))
    images = tuple(m * grid_n + n for m, n in coordinates)
    permutation = FinitePermutation(images)
    jumps = [
        abs(next_m - current_m) + abs(next_n - current_n)
        for (current_m, current_n), (next_m, next_n) in pairwise(coordinates)
    ]
    locality = sum(jump * jump for jump in jumps) / max(len(jumps), 1)
    resolved = resolve_schedule(pattern, grid_m, grid_n)
    description_bits = _DESCRIPTION_BITS[resolved]
    # Minimum-description-length proxy: a short decoder is preferred unless
    # a more expressive traversal materially reduces cache-distance jumps.
    score = locality + description_bits / 256.0
    symmetry_group = GridSymmetryGroup(grid_m, grid_n)
    return ScheduleCost(
        pattern=resolved,
        description_bits=description_bits,
        locality_cost=locality,
        worst_jump=max(jumps, default=0),
        score=score,
        permutation=permutation,
        canonical_signature=symmetry_group.canonicalize(permutation),
    )


def rank_schedules(grid_m: int, grid_n: int) -> list[ScheduleCost]:
    """Rank inequivalent, valid schedule representations by an MDL proxy."""
    representatives = {}
    for pattern in sorted(SCHEDULES - {"auto"}):
        cost = analyze_schedule(pattern, grid_m, grid_n)
        incumbent = representatives.get(cost.canonical_signature)
        if incumbent is None or (cost.score, cost.description_bits, cost.pattern) < (
            incumbent.score,
            incumbent.description_bits,
            incumbent.pattern,
        ):
            representatives[cost.canonical_signature] = cost
    ranked = list(representatives.values())
    return sorted(ranked, key=lambda cost: (cost.score, cost.description_bits, cost.pattern))


def optimize_tile_schedules(function: mir.MFunction) -> mir.MFunction:
    """Resolve auto schedules and fallback cases to branch-free IR."""
    for op in function.ops:
        if not isinstance(op, mir.MTileSchedule) or op.grid_m is None or op.grid_n is None:
            continue
        if op.pattern == "auto":
            op.pattern = rank_schedules(op.grid_m, op.grid_n)[0].pattern
        else:
            op.pattern = analyze_schedule(op.pattern, op.grid_m, op.grid_n).pattern
        op.is_static = True
    return function
