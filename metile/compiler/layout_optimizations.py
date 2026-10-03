"""Simplify proven layout identities without changing logical tensor values."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass, replace

from metile.ir import tile_ir as tir
from metile.ir.ownership import ThreadLayout
from metile.ir.types import TileType


@dataclass(frozen=True)
class LayoutRewrite:
    kind: str
    values: tuple[str, ...]
    replacement: str
    reason: str


@dataclass(frozen=True)
class LayoutOptimizationReport:
    conversions_before: int = 0
    conversions_after: int = 0
    identities_removed: int = 0
    inverse_pairs_cancelled: int = 0
    chains_composed: int = 0
    rewrites: tuple[LayoutRewrite, ...] = ()
    skipped: tuple[str, ...] = ()

    @property
    def conversions_removed(self) -> int:
        return self.conversions_before - self.conversions_after

    def to_dict(self) -> dict:
        return {**asdict(self), "conversions_removed": self.conversions_removed}


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


def _effective_layout(value: tir.Value) -> ThreadLayout | None:
    if not isinstance(value.type, TileType) or len(value.type.shape) != 1:
        return None
    if value.type.layout is not None:
        return value.type.layout
    try:
        return ThreadLayout.identity(value.type.numel)
    except ValueError:
        return None


def _geometry(layout: ThreadLayout) -> tuple[int, int, int]:
    return layout.size, layout.size // layout.elements_per_thread, layout.elements_per_thread


def _conversion_layouts(operation: tir.ConvertLayout):
    source = _effective_layout(operation.value)
    destination = operation.layout
    result = operation.result
    if (
        source is None
        or not isinstance(destination, ThreadLayout)
        or result is None
        or not isinstance(result.type, TileType)
        or result.type.layout != destination
        or result.type.shape != operation.value.type.shape
        or result.type.dtype != operation.value.type.dtype
        or _geometry(source) != _geometry(destination)
    ):
        return None
    if source != destination and (
        source.elements_per_thread != 1
        or operation.value.type.dtype not in {"f32", "f16", "i32", "u32", "bool"}
    ):
        return None
    return source, destination


def _resolve(value: tir.Value, aliases: dict[int, tir.Value]) -> tir.Value:
    while id(value) in aliases:
        value = aliases[id(value)]
    return value


def _rewrite_references(item, aliases: dict[int, tir.Value], memo: dict[int, object]):
    if isinstance(item, tir.Value):
        return _resolve(item, aliases)
    if id(item) in memo:
        return memo[id(item)]
    memo[id(item)] = item
    if isinstance(item, list):
        item[:] = [_rewrite_references(value, aliases, memo) for value in item]
    elif isinstance(item, tuple):
        rewritten = tuple(_rewrite_references(value, aliases, memo) for value in item)
        if any(new is not old for new, old in zip(rewritten, item, strict=True)):
            memo[id(item)] = rewritten
            return rewritten
    elif isinstance(item, dict):
        for key, value in item.items():
            item[key] = _rewrite_references(value, aliases, memo)
    elif is_dataclass(item):
        changes = {}
        for definition in fields(item):
            if isinstance(item, tir.Op) and definition.name == "result":
                continue
            value = getattr(item, definition.name)
            rewritten = _rewrite_references(value, aliases, memo)
            if rewritten is not value:
                changes[definition.name] = rewritten
        if item.__dataclass_params__.frozen and changes:
            rewritten = replace(item, **changes)
            memo[id(item)] = rewritten
            return rewritten
        for name, value in changes.items():
            setattr(item, name, value)
    return item


def _value_uses(item, counts: Counter[int], active: set[int]):
    if isinstance(item, tir.Value):
        counts[id(item)] += 1
        return
    if id(item) in active:
        return
    active.add(id(item))
    if isinstance(item, (tuple, list)):
        for value in item:
            _value_uses(value, counts, active)
    elif isinstance(item, dict):
        for value in item.values():
            _value_uses(value, counts, active)
    elif is_dataclass(item):
        for definition in fields(item):
            if isinstance(item, tir.Op) and definition.name == "result":
                continue
            _value_uses(getattr(item, definition.name), counts, active)
    active.remove(id(item))


def _remove_unused_conversions(function: tir.Function, candidates: set[int]):
    while candidates:
        uses = Counter()
        _value_uses(function, uses, set())
        removable = {
            id(operation)
            for operation in _walk(function.ops)
            if id(operation) in candidates and uses[id(operation.result)] == 0
        }
        if not removable:
            break

        def remove(operations, removals=removable):
            kept = []
            for operation in operations:
                if id(operation) in removals:
                    continue
                if hasattr(operation, "body"):
                    operation.body = remove(operation.body, removals)
                kept.append(operation)
            return kept

        function.ops = remove(function.ops)
        candidates.difference_update(removable)


def optimize_layout_conversions(
    function: tir.Function,
) -> tuple[tir.Function, LayoutOptimizationReport]:
    """Copy and simplify adjacent pure conversions after ownership validation.

    Unchanged functions are returned by identity. A round-trip rewrite removes
    the restored value's communication path; an intermediate conversion remains
    when another operation or tensor declaration still references its result.
    Nonidentity multi-register conversions remain outside the supported proof.
    """
    before = sum(isinstance(operation, tir.ConvertLayout) for operation in _walk(function.ops))
    if not before:
        return function, LayoutOptimizationReport()
    optimized = copy.deepcopy(function)
    aliases: dict[int, tir.Value] = {}
    candidates: set[int] = set()
    rewrites = []
    skipped = []

    def simplify(operations):
        kept = []
        for operation in operations:
            _rewrite_references(operation, aliases, {})
            if hasattr(operation, "body"):
                operation.body = simplify(operation.body)
            if not isinstance(operation, tir.ConvertLayout):
                kept.append(operation)
                continue
            layouts = _conversion_layouts(operation)
            if layouts is None:
                name = operation.result.name if operation.result is not None else "<missing>"
                skipped.append(f"{name}: unsupported or inconsistent conversion geometry/type")
                kept.append(operation)
                continue
            source, destination = layouts
            if source == destination:
                aliases[id(operation.result)] = operation.value
                rewrites.append(
                    LayoutRewrite(
                        "identity",
                        (operation.result.name,),
                        operation.value.name,
                        "source and destination have identical logical ownership and geometry",
                    )
                )
                continue
            previous = kept[-1] if kept else None
            if (
                not isinstance(previous, tir.ConvertLayout)
                or operation.value is not previous.result
                or _conversion_layouts(previous) is None
            ):
                kept.append(operation)
                continue
            original = previous.value
            candidates.add(id(previous))
            values = (previous.result.name, operation.result.name)
            if _effective_layout(original) == destination:
                aliases[id(operation.result)] = original
                rewrites.append(
                    LayoutRewrite(
                        "inverse",
                        values,
                        original.name,
                        "adjacent pure conversions restore every original logical element",
                    )
                )
            else:
                operation.value = original
                rewrites.append(
                    LayoutRewrite(
                        "compose",
                        values,
                        operation.result.name,
                        "adjacent bijective redistributions compose with matching shape and dtype",
                    )
                )
                kept.append(operation)
        return kept

    optimized.ops = simplify(optimized.ops)
    if not rewrites:
        return function, LayoutOptimizationReport(before, before, skipped=tuple(skipped))
    _rewrite_references(optimized, aliases, {})
    _remove_unused_conversions(optimized, candidates)
    after = sum(isinstance(operation, tir.ConvertLayout) for operation in _walk(optimized.ops))
    report = LayoutOptimizationReport(
        before,
        after,
        sum(rewrite.kind == "identity" for rewrite in rewrites),
        sum(rewrite.kind == "inverse" for rewrite in rewrites),
        sum(rewrite.kind == "compose" for rewrite in rewrites),
        tuple(rewrites),
        tuple(skipped),
    )
    return optimized, report
