from dataclasses import dataclass

from metile.ir.ownership import ThreadLayout


@dataclass(frozen=True)
class ScalarType:
    dtype: str  # "f32", "f16", "bf16", "i32", "u32", "u8", "bool"

    def to_msl(self) -> str:
        return {
            "f32": "float",
            "f16": "half",
            "bf16": "bfloat",
            "i32": "int",
            "u32": "uint",
            "u8": "uchar",
            "bool": "bool",
        }[self.dtype]

    def __repr__(self):
        return self.dtype


@dataclass(frozen=True)
class VectorType:
    dtype: str
    width: int = 4

    def __post_init__(self):
        if self.dtype not in {"f16", "f32"}:
            raise ValueError("vector memory values require f16 or f32 elements")
        if type(self.width) is not int or self.width != 4:
            raise ValueError("vector memory values require exactly four elements")

    def to_msl(self) -> str:
        return f"{ScalarType(self.dtype).to_msl()}{self.width}"

    def __repr__(self):
        return f"vector<{self.width}, {self.dtype}>"


@dataclass(frozen=True)
class TileType:
    shape: tuple[int, ...]
    dtype: str
    layout: ThreadLayout | None = None

    def __post_init__(self):
        if self.layout is not None:
            if not isinstance(self.layout, ThreadLayout):
                raise TypeError("tile execution layout must be a ThreadLayout")
            if self.shape != (self.layout.size,):
                raise ValueError("thread layout requires a one-dimensional tile of matching size")

    @property
    def numel(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    def to_msl(self) -> str:
        return ScalarType(self.dtype).to_msl()

    def __repr__(self):
        shape_str = "x".join(str(s) for s in self.shape)
        ownership = f", layout={self.layout!r}" if self.layout is not None else ""
        return f"tile<{shape_str}, {self.dtype}{ownership}>"


def merge_tile_layouts(*types) -> ThreadLayout | None:
    """Check pointwise ownership while preserving unconstrained legacy tile types."""
    tiles = [value_type for value_type in types if isinstance(value_type, TileType)]
    explicit = [value_type.layout for value_type in tiles if value_type.layout is not None]
    if not explicit:
        return None
    layout = explicit[0]
    if any(value_type.shape != (layout.size,) for value_type in tiles):
        raise ValueError("thread layouts require matching one-dimensional tile shapes")
    if any(value_type.layout is not None and value_type.layout != layout for value_type in tiles):
        raise ValueError("tile thread layouts differ; use convert_layout before combining values")
    if any(value_type.layout is None for value_type in tiles) and (
        layout.elements_per_thread != 1 or layout != ThreadLayout.identity(layout.size)
    ):
        raise ValueError("tile thread layouts differ; use convert_layout before combining values")
    return layout


@dataclass(frozen=True)
class PtrType:
    dtype: str
    address_space: str = "device"  # "device", "threadgroup", "constant"

    def to_msl(self) -> str:
        base = ScalarType(self.dtype).to_msl()
        return (
            f"device const {base}*"
            if self.address_space == "device"
            else f"{self.address_space} {base}*"
        )

    def to_msl_mut(self) -> str:
        base = ScalarType(self.dtype).to_msl()
        return f"device {base}*"

    def __repr__(self):
        return f"ptr<{self.dtype}>"


# Common types
I32 = ScalarType("i32")
U32 = ScalarType("u32")
BOOL = ScalarType("bool")
