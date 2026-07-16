from __future__ import annotations

from collections.abc import Iterator

SCHEDULES = frozenset({"auto", "diagonal", "hilbert", "linear", "morton"})

# A 4x4 Hilbert curve packed as two bits per coordinate. The same constants
# are emitted in MSL so the reference implementation and GPU mapping agree.
HILBERT_4X4_M = 0xEBFA5014
HILBERT_4X4_N = 0x05BEBE50


def validate_schedule(pattern: str, block_size: int | None = None) -> str:
    """Validate and normalize a tile schedule name."""
    normalized = pattern or "linear"
    if normalized not in SCHEDULES:
        choices = ", ".join(sorted(SCHEDULES))
        raise ValueError(f"Unknown tile schedule '{pattern}'; expected one of: {choices}")
    if block_size is not None:
        expected = 4 if normalized == "hilbert" else 2
        if normalized in {"hilbert", "morton"} and block_size != expected:
            raise ValueError(f"{normalized} scheduling requires block_size={expected}")
    return normalized


def resolve_schedule(pattern: str, grid_m: int, grid_n: int) -> str:
    """Select a bijective schedule supported by the concrete grid shape."""
    normalized = validate_schedule(pattern)
    if normalized in {"auto", "hilbert"} and grid_m % 4 == 0 and grid_n % 4 == 0:
        return "hilbert"
    if normalized in {"auto", "hilbert", "morton"} and grid_m % 2 == 0 and grid_n % 2 == 0:
        return "morton"
    if normalized in {"auto", "hilbert", "morton"}:
        return "diagonal"
    return normalized


def schedule_coordinate(
    linear_id: int, grid_m: int, grid_n: int, pattern: str = "auto"
) -> tuple[int, int]:
    """Reference tile mapping used by schedule correctness tests."""
    if linear_id < 0 or linear_id >= grid_m * grid_n:
        raise IndexError("tile id is outside the dispatch grid")

    schedule = resolve_schedule(pattern, grid_m, grid_n)
    if schedule == "hilbert":
        panels_n = grid_n // 4
        panel_id, within = divmod(linear_id, 16)
        panel_m, panel_n = divmod(panel_id, panels_n)
        local_m = (HILBERT_4X4_M >> (within * 2)) & 3
        local_n = (HILBERT_4X4_N >> (within * 2)) & 3
        return panel_m * 4 + local_m, panel_n * 4 + local_n
    if schedule == "morton":
        panels_n = grid_n // 2
        panel_id, within = divmod(linear_id, 4)
        panel_m, panel_n = divmod(panel_id, panels_n)
        return panel_m * 2 + within // 2, panel_n * 2 + within % 2

    pid_m, pid_n = divmod(linear_id, grid_n)
    if schedule == "diagonal":
        pid_n = (pid_n + pid_m) % grid_n
    return pid_m, pid_n


def schedule_coordinates(
    grid_m: int, grid_n: int, pattern: str = "auto"
) -> Iterator[tuple[int, int]]:
    """Yield every coordinate in GPU dispatch order."""
    for linear_id in range(grid_m * grid_n):
        yield schedule_coordinate(linear_id, grid_m, grid_n, pattern)
