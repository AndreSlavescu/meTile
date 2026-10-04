"""Checked ownership of logical tile elements by physical GPU threads."""

from dataclasses import dataclass

_REGISTER_COUNTS = {1, 2, 4, 8, 16, 32}


@dataclass(frozen=True)
class ThreadLayout:
    """A bijection from thread/register coordinates to logical tile indices.

    Physical coordinates pack thread bits first, followed by register bits.
    ``bit_order[logical_bit]`` selects the packed physical bit placed at that
    logical bit position. ``xor_mask`` is applied to the resulting logical index.
    Every physical thread owns a power-of-two number of scalar elements, up to 32.
    """

    bit_order: tuple[int, ...]
    xor_mask: int = 0
    elements_per_thread: int = 1

    def __post_init__(self):
        if (
            type(self.elements_per_thread) is not int
            or self.elements_per_thread not in _REGISTER_COUNTS
        ):
            raise ValueError("elements_per_thread must be a power of two between 1 and 32")
        register_bits = self.elements_per_thread.bit_length() - 1
        if (
            not isinstance(self.bit_order, tuple)
            or not 5 + register_bits <= len(self.bit_order) <= 10 + register_bits
            or any(type(bit) is not int for bit in self.bit_order)
            or sorted(self.bit_order) != list(range(len(self.bit_order)))
        ):
            raise ValueError(
                "bit_order must be an integer bit permutation describing 32 to 1024 threads"
            )
        if type(self.xor_mask) is not int or not 0 <= self.xor_mask < self.size:
            raise ValueError("xor_mask must be an integer between 0 and layout size minus one")

    @property
    def size(self) -> int:
        return 1 << len(self.bit_order)

    @property
    def thread_count(self) -> int:
        return self.size // self.elements_per_thread

    @classmethod
    def identity(cls, size: int, *, elements_per_thread: int = 1) -> "ThreadLayout":
        """Construct identity ownership with 32 to 1024 physical threads.

        ``size`` counts logical elements and must be between 32 and 1024 times
        ``elements_per_thread``. Each register slot spans all physical threads.
        """
        if type(elements_per_thread) is not int or elements_per_thread not in _REGISTER_COUNTS:
            raise ValueError("elements_per_thread must be a power of two between 1 and 32")
        if (
            type(size) is not int
            or not 32 * elements_per_thread <= size <= 1024 * elements_per_thread
            or size & (size - 1)
        ):
            raise ValueError(
                "layout size must be a power of two spanning 32 to 1024 physical threads"
            )
        return cls(tuple(range(size.bit_length() - 1)), elements_per_thread=elements_per_thread)

    def logical_index(self, thread: int, register: int = 0) -> int:
        """Return the logical element held in a physical thread's register slot."""
        if type(thread) is not int or not 0 <= thread < self.thread_count:
            raise ValueError("thread index must be an integer within the layout")
        if type(register) is not int or not 0 <= register < self.elements_per_thread:
            raise ValueError("register index must be an integer within the layout")
        packed = thread + register * self.thread_count
        return self.xor_mask ^ sum(
            ((packed >> source_bit) & 1) << logical_bit
            for logical_bit, source_bit in enumerate(self.bit_order)
        )

    def owner(self, index: int) -> int:
        """Return the physical thread owning logical element ``index``."""
        return self._physical_index(index) % self.thread_count

    def register(self, index: int) -> int:
        """Return the physical register slot holding logical element ``index``."""
        return self._physical_index(index) // self.thread_count

    def _physical_index(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < self.size:
            raise ValueError("logical index must be an integer within the layout")
        unmasked = index ^ self.xor_mask
        return sum(
            ((unmasked >> logical_bit) & 1) << source_bit
            for logical_bit, source_bit in enumerate(self.bit_order)
        )
