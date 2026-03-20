from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarType:
    dtype: str  # "f32", "f16", "bf16", "i32", "u32", "bool"

    def to_msl(self) -> str:
        return {
            "f32": "float",
            "f16": "half",
            "bf16": "bfloat",
            "i32": "int",
            "u32": "uint",
            "bool": "bool",
        }[self.dtype]

    def __repr__(self):
        return self.dtype


@dataclass(frozen=True)
class TileType:
    shape: tuple[int, ...]
    dtype: str

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
        return f"tile<{shape_str}, {self.dtype}>"


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
