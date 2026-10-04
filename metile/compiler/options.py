"""Checked execution constraints; unspecified decisions remain compiler selected."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Schedule:
    """Requirements on a kernel schedule, not hints the compiler may ignore.

    Pass as ``SCHEDULE=Schedule(...)`` when compiling or preparing a kernel.
    Staging describes compiler-created matrix temporaries, not tensor view storage.
    """

    backend: str = "auto"
    num_simdgroups: int | None = None
    vector_width: int | None = None
    staging: str = "auto"
    double_buffer: bool | None = None

    def __post_init__(self):
        if self.backend not in {"auto", "simdgroup", "tensor_ops", "nax", "elementwise"}:
            raise ValueError(
                "schedule backend must be auto, simdgroup, tensor_ops, nax or elementwise"
            )
        if self.staging not in {"auto", "device", "threadgroup"}:
            raise ValueError("schedule staging must be auto, device or threadgroup")
        if self.num_simdgroups is not None and (
            isinstance(self.num_simdgroups, bool)
            or not isinstance(self.num_simdgroups, int)
            or not 1 <= self.num_simdgroups <= 32
        ):
            raise ValueError("num_simdgroups must be an integer from 1 to 32, or None")
        if self.vector_width is not None and (
            isinstance(self.vector_width, bool)
            or not isinstance(self.vector_width, int)
            or self.vector_width not in {1, 4}
        ):
            raise ValueError("vector_width must be 1, 4 or None")
        if self.double_buffer is not None and not isinstance(self.double_buffer, bool):
            raise ValueError("double_buffer must be True, False or None")
