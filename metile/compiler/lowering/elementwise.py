"""Lowering for element-wise kernels."""

from __future__ import annotations

from metile.compiler.lowering.common import (
    _MSL_TYPES,
    LoweringError,
)
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType


class _ElementwiseLoweringContext:
    def __init__(self, func: tir.Function):
        self.func = func
        self.mfunc = mir.MFunction(name=f"mtile_{func.name}")
        # Map from Tile IR Value -> Metal IR MValue
        self.value_map: dict[str, mir.MValue] = {}
        # Track which values are tile-expanded (use thread_position_in_grid)
        self.tid_value: mir.MValue | None = None
        # Track the block size for grid computation
        self.block_size: int | None = None
        # Track scalar param values by name -> MValue
        self.param_values: dict[str, mir.MValue] = {}
        # Row-parallel mode values
        self.tgp_id: mir.MValue | None = None
        self.lid_value: mir.MValue | None = None
        self.sgid: mir.MValue | None = None
        self.slid: mir.MValue | None = None
        # Lowering mode: "elementwise", "row_wise", "row_parallel"
        self._mode: str = "elementwise"
        self._reduce_counter: int = 0
        self._acc_counter: int = 0
        self._next_sg: int = 0  # tracks next available simdgroup index for role assignment
        # Track shared memory allocations: tile IR value name -> threadgroup array name
        self._shared_allocs: dict[str, str] = {}
        self._loop_post_refs: dict[int, set[str]] = {}
        self._index_loop_post_refs(self.func.ops)

    def lower(self) -> mir.MFunction:
        self._lower_params()

        if self._has_reduce() or self._has_shared():
            self._mode = "row_parallel"
            self._setup_row_parallel()
        elif self._has_arange():
            self._mode = "elementwise"
            self._setup_elementwise()
        else:
            self._mode = "row_wise"
            self._setup_row_wise()

        self._lower_ops()
        self._set_grid()
        return self.mfunc

    def _setup_elementwise(self):
        """Standard element-wise: tid = pid * BLOCK + arange."""
        tid_op = mir.ThreadPositionInGrid(axis=0)
        self.tid_value = self.mfunc.add_op(tid_op, "tid")

    def _setup_row_wise(self):
        """Row-wise: one thread per row, no arange."""
        tid_op = mir.ThreadPositionInGrid(axis=0)
        self.tid_value = self.mfunc.add_op(tid_op, "tid")

    def _setup_row_parallel(self):
        """Row-parallel: multiple threads per row with reduction."""
        self.tgp_id = self.mfunc.add_op(mir.ThreadgroupPositionInGrid(axis=0), "tgp_id_x")
        self.lid_value = self.mfunc.add_op(mir.ThreadPositionInThreadgroup(axis=0), "lid")
        self.sgid = self.mfunc.add_op(mir.MSimdgroupId(), "sgid")
        self.slid = self.mfunc.add_op(mir.MThreadInSimdgroup(), "slid")
        self.tid_value = None  # not used in row-parallel

    def _lower_params(self):
        for p in self.func.params:
            if isinstance(p.type, PtrType):
                mp = mir.MParam(
                    name=p.name,
                    type=p.type,
                    is_output=p.is_output,
                    is_scalar=False,
                )
            elif isinstance(p.type, ScalarType):
                mp = mir.MParam(
                    name=p.name,
                    type=p.type,
                    is_scalar=True,
                )
            else:
                raise LoweringError(f"Unsupported param type: {p.type}")
            self.mfunc.params.append(mp)
            # Create a sentinel MValue for param references
            mv = mir.MValue(p.name, p.type)
            self.param_values[p.name] = mv
            self.value_map[p.name] = mv

    def _lower_ops(self):
        # Collect all ops that are inside if-blocks (stores with masks)
        # First pass: identify mask values and the if-block pattern
        mask_value = None

        # Analyze: find the mask and split ops
        for op in self.func.ops:
            if isinstance(op, tir.Store) and op.mask is not None:
                # This store is masked - everything from the mask def onward
                # goes inside an if-block
                mask_value = op.mask
                break

        # Second pass: lower ops, grouping masked stores into if-block
        body_ops = []
        for op in self.func.ops:
            lowered = self._lower_op(op)
            if lowered is not None:
                for m_op in lowered:
                    body_ops.append(m_op)

        # If we have a mask, wrap the relevant ops in an if-block
        if mask_value and mask_value.name in self.value_map:
            mask_mv = self.value_map[mask_value.name]
            # Find the compare op that produces the mask
            # Everything after (and including loads) goes inside the if
            compare_idx = None
            for i, m_op in enumerate(body_ops):
                if hasattr(m_op, "result") and m_op.result and m_op.result is mask_mv:
                    compare_idx = i
                    break

            if compare_idx is not None:
                pre_ops = body_ops[: compare_idx + 1]  # include the compare
                post_ops = body_ops[compare_idx + 1 :]  # loads, compute, stores
                if_block = mir.IfBlock(condition=mask_mv, body=post_ops)
                for m_op in pre_ops:
                    self.mfunc.ops.append(m_op)
                self.mfunc.ops.append(if_block)
            else:
                for m_op in body_ops:
                    self.mfunc.ops.append(m_op)
        else:
            for m_op in body_ops:
                self.mfunc.ops.append(m_op)

    def _lower_op(self, op: tir.Op) -> list[mir.MOp] | None:
        """Lower a single Tile IR op to Metal IR op(s). Returns list or None."""
        if isinstance(op, tir.ProgramId):
            if self._mode == "row_parallel":
                self.value_map[op.result.name] = self.tgp_id
            elif self._mode == "elementwise":
                self.value_map[op.result.name] = None  # placeholder for pid*BLOCK+arange
            else:  # row_wise
                self.value_map[op.result.name] = self.tid_value
            return None

        elif isinstance(op, tir.Constant):
            m_op = mir.MConstant(value=op.value, dtype=op.dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.Cast):
            value = self._resolve(op.value)
            m_op = mir.MCast(value=value, target_dtype=op.dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.Arange):
            self.block_size = op.size
            if self._mode == "row_parallel":
                self.value_map[op.result.name] = self.lid_value
            else:
                self.value_map[op.result.name] = self.tid_value
            return None

        elif isinstance(op, tir.Reduce):
            return self._lower_reduce(op)

        elif isinstance(op, tir.BinOp):
            return self._lower_binop(op)

        elif isinstance(op, tir.Compare):
            return self._lower_compare(op)

        elif isinstance(op, tir.Load):
            return self._lower_load(op)

        elif isinstance(op, tir.Store):
            return self._lower_store(op)

        elif isinstance(op, tir.Unary):
            return self._lower_unary(op)

        elif isinstance(op, tir.Select):
            return self._lower_select(op)

        elif isinstance(op, tir.SharedAlloc):
            msl_type = _MSL_TYPES.get(op.dtype, "float")
            alloc_name = op.result.name  # e.g., "shared_5"
            alloc_op = mir.MThreadgroupAlloc(
                alloc_name=alloc_name, elem_type=msl_type, size=op.size
            )
            self._shared_allocs[op.result.name] = alloc_name
            # Create a sentinel MValue for the shared memory pointer
            val = mir.MValue(alloc_name, PtrType(op.dtype, "threadgroup"))
            self.value_map[op.result.name] = val
            return [alloc_op]

        elif isinstance(op, tir.Barrier):
            return [mir.MBarrier(kind="threadgroup", flags="mem_threadgroup")]

        elif isinstance(op, tir.ThreadId):
            # Ensure lid is available
            if self.lid_value is None:
                self.lid_value = self.mfunc.add_op(mir.ThreadPositionInThreadgroup(axis=0), "lid")
            val = self.lid_value
            self.value_map[op.result.name] = val
            return None

        elif isinstance(op, tir.PtrOffset):
            # ptr + offsets: track as (base_ptr, index) tuple
            ptr_val = self.value_map.get(op.ptr.name)
            offset_val = self.value_map.get(op.offsets.name, self.tid_value)

            if isinstance(ptr_val, tuple) and len(ptr_val) == 2:
                # Chained PtrOffset (e.g. X + a + b): combine offsets
                base, old_offset = ptr_val
                if isinstance(old_offset, mir.MValue) and isinstance(offset_val, mir.MValue):
                    ops = []
                    rhs = offset_val
                    if old_offset.type != offset_val.type:
                        cast_op = mir.MCast(value=offset_val, target_dtype=old_offset.type.dtype)
                        cast_v = mir.MValue(
                            f"cast_off_{op.result.name}", cast_op.result_type(), cast_op
                        )
                        cast_op.result = cast_v
                        ops.append(cast_op)
                        rhs = cast_v
                    combined_op = mir.MBinOp(op="add", lhs=old_offset, rhs=rhs)
                    combined_mv = mir.MValue(
                        f"_off_{op.result.name}", combined_op.result_type(), combined_op
                    )
                    combined_op.result = combined_mv
                    ops.append(combined_op)
                    self.value_map[op.result.name] = (base, combined_mv)
                    return ops
                else:
                    self.value_map[op.result.name] = (base, offset_val)
                    return None
            else:
                self.value_map[op.result.name] = (ptr_val, offset_val)
                return None

        elif isinstance(op, tir.ForRange):
            return self._lower_for_range(op)

        elif isinstance(op, tir.SimdShuffleXor):
            val = self._resolve(op.value)
            mask = self._resolve(op.mask)
            dtype = "f32"
            if isinstance(op.value.type, ScalarType):
                dtype = op.value.type.dtype
            m_op = mir.MSimdShuffleXor(value=val, mask=mask, dtype=dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.SimdBroadcast):
            val = self._resolve(op.value)
            lane = self._resolve(op.lane)
            dtype = "f32"
            if isinstance(op.value.type, ScalarType):
                dtype = op.value.type.dtype
            m_op = mir.MSimdBroadcast(value=val, lane=lane, dtype=dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.SimdLaneId):
            m_op = mir.MThreadInSimdgroup()
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.SimdgroupRole):
            return self._lower_simdgroup_role(op)

        elif isinstance(op, tir.TileSwizzle):
            # TileSwizzle is metadata-only — the swizzle_pattern is read from
            # func.swizzle_pattern during GEMM lowering. Skip in elementwise.
            return None

        elif isinstance(op, tir.Zeros):
            # Zeros is consumed implicitly by the GEMM accumulator path.
            # In elementwise context, lower to a zero constant.
            m_op = mir.MConstant(value=0, dtype=op.dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, (tir.Dot, tir.TileLoad, tir.TileStore, tir.PersistentRange)):
            # These ops are handled by the specialized GEMM/persistent lowering
            # paths, not the elementwise _lower_op dispatch.
            raise LoweringError(
                f"{type(op).__name__} encountered in elementwise lowering — "
                f"this op requires a GEMM kernel context"
            )

        else:
            raise LoweringError(f"Unsupported Tile IR op: {type(op).__name__}")

    def _lower_binop(self, op: tir.BinOp) -> list[mir.MOp] | None:
        lhs = self.value_map.get(op.lhs.name)
        rhs = self.value_map.get(op.rhs.name)

        # Check if this is the pid * BLOCK + arange pattern
        # pid * BLOCK: lhs is None (program_id placeholder), rhs is BLOCK constant
        if lhs is None and isinstance(op.lhs.defining_op, tir.ProgramId):
            # pid * BLOCK -> we still map to tid when combined with arange
            # For now, store a marker
            self.value_map[op.result.name] = ("pid_times_block",)
            return None

        # pid * BLOCK + arange -> thread_position_in_grid
        if (
            isinstance(lhs, tuple)
            and len(lhs) == 1
            and lhs[0] == "pid_times_block"
            and rhs is self.tid_value
        ):
            self.value_map[op.result.name] = self.tid_value
            return None

        # Regular scalar binary op
        if isinstance(lhs, mir.MValue) and isinstance(rhs, mir.MValue):
            # May need a cast if types differ (e.g., uint vs int)
            ops = []
            if lhs.type != rhs.type:
                cast_op = mir.MCast(value=rhs, target_dtype=lhs.type.dtype)
                cast_v = mir.MValue(f"cast_{op.result.name}", cast_op.result_type(), cast_op)
                cast_op.result = cast_v
                ops.append(cast_op)
                rhs = cast_v

            m_op = mir.MBinOp(op=op.op, lhs=lhs, rhs=rhs)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            ops.append(m_op)
            return ops

        # Tile-level binop where one side is tid and other is a scalar/tile
        if lhs is self.tid_value or rhs is self.tid_value:
            actual_lhs = lhs if isinstance(lhs, mir.MValue) else self.tid_value
            actual_rhs = rhs if isinstance(rhs, mir.MValue) else self.tid_value
            m_op = mir.MBinOp(op=op.op, lhs=actual_lhs, rhs=actual_rhs)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        raise LoweringError(f"Cannot lower BinOp: lhs={lhs}, rhs={rhs}, op={op.op}")

    def _lower_compare(self, op: tir.Compare) -> list[mir.MOp]:
        lhs = self._resolve(op.lhs)
        rhs = self._resolve(op.rhs)
        ops = []

        # Cast if types differ
        if lhs.type != rhs.type:
            cast_op = mir.MCast(value=rhs, target_dtype=lhs.type.dtype)
            cast_v = mir.MValue(f"cast_{op.result.name}", cast_op.result_type(), cast_op)
            cast_op.result = cast_v
            ops.append(cast_op)
            rhs = cast_v

        m_op = mir.MCompare(predicate=op.predicate, lhs=lhs, rhs=rhs)
        mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
        m_op.result = mv
        self.value_map[op.result.name] = mv
        ops.append(m_op)
        return ops

    def _is_shared_ptr(self, base: mir.MValue) -> bool:
        """Check if a base MValue corresponds to a shared memory allocation."""
        return isinstance(base, mir.MValue) and base.name in self._shared_allocs

    def _lower_load(self, op: tir.Load) -> list[mir.MOp]:
        ptr_info = self.value_map.get(op.ptr.name)
        # ptr_info could be a tuple (base_ptr, index) from PtrOffset
        if isinstance(ptr_info, tuple) and len(ptr_info) == 2:
            base, index = ptr_info
        else:
            base = self._resolve_ptr(op.ptr)
            index = self._resolve(op.offsets)

        dtype = op.ptr.type.dtype if isinstance(op.ptr.type, PtrType) else "f32"

        # Shared memory load
        if self._is_shared_ptr(base):
            array_name = self._shared_allocs[base.name]
            m_op = mir.MThreadgroupLoad(array_name=array_name, index=index, dtype=dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        m_op = mir.DeviceLoad(ptr=base, index=index, dtype=dtype)
        mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
        m_op.result = mv
        self.value_map[op.result.name] = mv
        return [m_op]

    def _lower_store(self, op: tir.Store) -> list[mir.MOp]:
        ptr_info = self.value_map.get(op.ptr.name)
        if isinstance(ptr_info, tuple) and len(ptr_info) == 2:
            base, index = ptr_info
        else:
            base = self._resolve_ptr(op.ptr)
            index = self._resolve(op.offsets)

        value = self._resolve(op.value)

        # Shared memory store
        if self._is_shared_ptr(base):
            array_name = self._shared_allocs[base.name]
            m_op = mir.MThreadgroupStore(array_name=array_name, index=index, value=value)
            return [m_op]

        m_op = mir.DeviceStore(ptr=base, index=index, value=value)
        return [m_op]

    def _lower_unary(self, op: tir.Unary) -> list[mir.MOp]:
        operand = self._resolve(op.operand)
        m_op = mir.MUnary(op=op.op, operand=operand)
        mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
        m_op.result = mv
        self.value_map[op.result.name] = mv
        return [m_op]

    def _lower_select(self, op: tir.Select) -> list[mir.MOp]:
        cond = self._resolve(op.condition)
        true_v = self._resolve(op.true_val)
        false_v = self._resolve(op.false_val)
        m_op = mir.MSelect(condition=cond, true_val=true_v, false_val=false_v)
        mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
        m_op.result = mv
        self.value_map[op.result.name] = mv
        return [m_op]

    def _resolve(self, val: tir.Value) -> mir.MValue:
        """Resolve a Tile IR value to its Metal IR equivalent."""
        mv = self.value_map.get(val.name)
        if isinstance(mv, mir.MValue):
            return mv
        if mv is self.tid_value:
            return self.tid_value
        # Could be a param
        if val.name in self.param_values:
            return self.param_values[val.name]
        raise LoweringError(f"Cannot resolve value: %{val.name}")

    def _resolve_ptr(self, val: tir.Value) -> mir.MValue:
        """Resolve a pointer value."""
        mv = self.value_map.get(val.name)
        if isinstance(mv, mir.MValue):
            return mv
        if val.name in self.param_values:
            return self.param_values[val.name]
        raise LoweringError(f"Cannot resolve pointer: %{val.name}")

    def _lower_simdgroup_role(self, op: tir.SimdgroupRole) -> list[mir.MOp]:
        """Lower a SimdgroupRole to MSimdgroupRoleBlock.

        Ensures sgid is available, computes simdgroup range for this role,
        and lowers the body ops into the role block.
        """
        # Ensure we have sgid available
        result_ops = []
        if self.sgid is None:
            self.sgid = self.mfunc.add_op(mir.MSimdgroupId(), "sgid")

        # Compute SG assignment using cumulative tracking
        tg_size = self.mfunc.threadgroup_size[0]
        total_sgs = tg_size // 32

        if op.num_sgs > 0:
            role_sgs = op.num_sgs
        else:
            role_sgs = total_sgs // op.num_roles

        first_sg = self._next_sg
        self._next_sg += role_sgs

        # Lower body ops
        body_metal_ops = []
        for body_op in op.body:
            lowered = self._lower_op(body_op)
            if lowered is not None:
                body_metal_ops.extend(lowered)

        role_block = mir.MSimdgroupRoleBlock(
            role=op.role,
            num_roles=op.num_roles,
            first_sg=first_sg,
            num_sgs=role_sgs,
            sgid=self.sgid,
            body=body_metal_ops,
        )
        result_ops.append(role_block)
        return result_ops

    def _lower_for_range(self, op: tir.ForRange) -> list[mir.MOp]:
        """Lower a ForRange (tile_range) to an MForLoop in element-wise context.

        Detects accumulation patterns (values defined inside the loop that are
        used after it) and emits MVarDecl/MVarAssign for loop-carried deps.
        In row-parallel mode, also detects masks and wraps body in IfBlock.
        """
        start = self._resolve(op.start)
        end = self._resolve(op.end)

        iv_mv = mir.MValue(op.iv.name, I32)
        self.value_map[op.iv.name] = iv_mv

        # Pre-pass: detect accumulation patterns in the Tile IR body
        acc_infos = self._detect_accumulation(op)

        # Pre-pass: detect mask in body (for row-parallel mode)
        body_mask_name = None
        if self._mode == "row_parallel":
            for body_op in op.body:
                if isinstance(body_op, tir.Load) and getattr(body_op, "mask", None) is not None:
                    body_mask_name = body_op.mask.name
                    break
                if isinstance(body_op, tir.Store) and getattr(body_op, "mask", None) is not None:
                    body_mask_name = body_op.mask.name
                    break

        # Lower body ops
        body_metal_ops = []
        for body_op in op.body:
            lowered = self._lower_op(body_op)
            if lowered is not None:
                body_metal_ops.extend(lowered)

        result_ops = []

        if acc_infos:
            for init_tile_name, init_const_value, final_tile_name in acc_infos:
                init_mv = self.value_map.get(init_tile_name)
                final_mv = self.value_map.get(final_tile_name)

                if isinstance(init_mv, mir.MValue) and isinstance(final_mv, mir.MValue):
                    acc_id = self._acc_counter
                    self._acc_counter += 1
                    var_name = f"_acc_{acc_id}"
                    dtype = init_mv.type.dtype if hasattr(init_mv.type, "dtype") else "f32"

                    # Create a new MConstant for the init value (before the loop)
                    init_const_op = mir.MConstant(value=init_const_value, dtype=dtype)
                    init_const_mv = mir.MValue(f"_acc_init_{acc_id}", ScalarType(dtype))
                    init_const_op.result = init_const_mv
                    init_const_mv.defining_op = init_const_op
                    result_ops.append(init_const_op)

                    # Declare the mutable accumulator variable
                    result_ops.append(
                        mir.MVarDecl(var_name=var_name, init_value=init_const_mv, dtype=dtype)
                    )

                    # Remove the MConstant from body_metal_ops (it's now before the loop)
                    body_metal_ops = [
                        m
                        for m in body_metal_ops
                        if not (isinstance(m, mir.MConstant) and m.result is init_mv)
                    ]

                    # Replace references to init_mv in body with the accumulator var
                    var_mv = mir.MValue(var_name, init_mv.type)
                    self._replace_mvalue_refs(body_metal_ops, init_mv, var_mv)

                    # Assign final value back to accumulator at end of body
                    body_metal_ops.append(mir.MVarAssign(var_name=var_name, value=final_mv))

                    # Map final value to accumulator for post-loop use
                    self.value_map[final_tile_name] = var_mv

        # Handle masking: wrap post-mask body ops in IfBlock
        if body_mask_name and body_mask_name in self.value_map:
            mask_mv = self.value_map[body_mask_name]
            if isinstance(mask_mv, mir.MValue):
                compare_idx = None
                for i, m_op in enumerate(body_metal_ops):
                    if isinstance(m_op, mir.MCompare) and m_op.result is mask_mv:
                        compare_idx = i
                        break
                if compare_idx is not None:
                    if op.masked_identity is None:
                        pre_ops = body_metal_ops[: compare_idx + 1]
                        if_body = body_metal_ops[compare_idx + 1 :]
                        body_metal_ops = [
                            *pre_ops,
                            mir.IfBlock(condition=mask_mv, body=if_body),
                        ]
                    else:
                        body_metal_ops = self._predicate_masked_load(
                            body_metal_ops, compare_idx, mask_mv, op.masked_identity
                        )

        loop = mir.MForLoop(
            iv_name=op.iv.name,
            start=start,
            end=end,
            step=op.step,
            body=body_metal_ops,
        )
        if op.num_stages > 1:
            loop._num_stages = op.num_stages
        result_ops.append(loop)
        return result_ops

    def _predicate_masked_load(self, body_metal_ops, compare_idx, mask_mv, identity):
        """Guard only the load, and give masked lanes a reduction identity.

        A threadgroup reduction inside a mask branch deadlocks or reads garbage, because
        threads whose lane is masked off never reach the barrier. So instead of wrapping
        the whole body, hoist the loaded value to a variable seeded with the identity,
        assign it under the branch, and leave everything downstream unguarded. Every
        thread then reaches the reduction, and masked lanes contribute a value the law
        says cannot change the result.
        """
        load_idx = None
        for index in range(compare_idx + 1, len(body_metal_ops)):
            if isinstance(body_metal_ops[index], mir.DeviceLoad):
                load_idx = index
                break
        if load_idx is None:
            pre_ops = body_metal_ops[: compare_idx + 1]
            return [
                *pre_ops,
                mir.IfBlock(condition=mask_mv, body=body_metal_ops[compare_idx + 1 :]),
            ]

        loaded = body_metal_ops[load_idx].result
        dtype = loaded.type.dtype if hasattr(loaded.type, "dtype") else "f32"
        var_name = f"_masked_{loaded.name}"

        seed_op = mir.MConstant(value=identity, dtype=dtype)
        seed_mv = mir.MValue(f"{var_name}_identity", ScalarType(dtype))
        seed_op.result = seed_mv
        seed_mv.defining_op = seed_op

        guarded = body_metal_ops[compare_idx + 1 : load_idx + 1]
        guarded.append(mir.MVarAssign(var_name=var_name, value=loaded))
        trailing = body_metal_ops[load_idx + 1 :]
        self._replace_mvalue_refs(trailing, loaded, mir.MValue(var_name, loaded.type))

        return [
            *body_metal_ops[: compare_idx + 1],
            seed_op,
            mir.MVarDecl(var_name=var_name, init_value=seed_mv, dtype=dtype, tile_valued=True),
            mir.IfBlock(condition=mask_mv, body=guarded),
            *trailing,
        ]

    def _detect_accumulation(self, for_range_op: tir.ForRange):
        """Detect accumulation patterns in a ForRange body.

        Looks for explicit scalar initializers that feed into BinOp chains,
        where the final values are used after the ForRange. Returns a list of
        (init_name, init_value, final_name) tuples, or None if empty.
        """
        body = for_range_op.body

        # Map body value names to their defining ops
        body_defs = {}
        for body_op in body:
            if hasattr(body_op, "result") and body_op.result:
                body_defs[body_op.result.name] = body_op

        # Find values used by later siblings in the loop's containing block.
        post_refs = self._loop_post_refs.get(id(for_range_op), set())

        # Find escaped values (defined in body, used after)
        escaped = [name for name in body_defs if name in post_refs]

        results = []
        used_inits = set()
        for esc_name in escaped:
            chain = self._walk_acc_chain(esc_name, body_defs)
            if chain and chain[0] not in used_inits:
                results.append(chain)
                used_inits.add(chain[0])
        return results or None

    def _walk_acc_chain(self, start_name, body_defs):
        """Walk backward through BinOp chains to find a scalar state root.

        Returns (init_name, init_value, final_name) or None.
        """
        current_name = start_name
        visited = set()

        while current_name in body_defs and current_name not in visited:
            visited.add(current_name)
            op = body_defs[current_name]

            if isinstance(op, tir.BinOp):
                for operand in (op.lhs, op.rhs):
                    defining_op = operand.defining_op
                    if isinstance(defining_op, tir.Constant) and defining_op.explicit_scalar:
                        return (operand.name, defining_op.value, start_name)

                lhs_name = op.lhs.name
                rhs_name = op.rhs.name

                # Check if either side is a Constant (the accumulation root)
                if lhs_name in body_defs and isinstance(body_defs[lhs_name], tir.Constant):
                    return (lhs_name, body_defs[lhs_name].value, start_name)
                if rhs_name in body_defs and isinstance(body_defs[rhs_name], tir.Constant):
                    return (rhs_name, body_defs[rhs_name].value, start_name)

                # Follow the accumulation chain (the side that's a BinOp)
                if lhs_name in body_defs and isinstance(body_defs[lhs_name], tir.BinOp):
                    current_name = lhs_name
                elif rhs_name in body_defs and isinstance(body_defs[rhs_name], tir.BinOp):
                    current_name = rhs_name
                else:
                    break
            else:
                break
        return None

    def _collect_tile_ir_refs(self, op, refs: set):
        """Collect value names referenced by a Tile IR op."""
        for attr in (
            "lhs",
            "rhs",
            "operand",
            "condition",
            "true_val",
            "false_val",
            "ptr",
            "offsets",
            "value",
        ):
            val = getattr(op, attr, None)
            if isinstance(val, tir.Value):
                refs.add(val.name)
        if isinstance(op, tir.Store) and getattr(op, "mask", None) is not None:
            refs.add(op.mask.name)
        for body_op in getattr(op, "body", ()):
            self._collect_tile_ir_refs(body_op, refs)

    def _index_loop_post_refs(self, ops):
        """Index references after every runtime loop within its parent block."""
        for index, op in enumerate(ops):
            if isinstance(op, tir.ForRange):
                refs = set()
                for sibling in ops[index + 1 :]:
                    self._collect_tile_ir_refs(sibling, refs)
                self._loop_post_refs[id(op)] = refs
            body = getattr(op, "body", None)
            if body is not None:
                self._index_loop_post_refs(body)

    def _replace_mvalue_refs(self, ops: list, old_mv: mir.MValue, new_mv: mir.MValue):
        """Replace all references to old_mv with new_mv in Metal IR ops."""
        for m_op in ops:
            for attr in (
                "lhs",
                "rhs",
                "operand",
                "condition",
                "true_val",
                "false_val",
                "ptr",
                "index",
                "value",
            ):
                if getattr(m_op, attr, None) is old_mv:
                    setattr(m_op, attr, new_mv)

    def _lower_reduce(self, op: tir.Reduce) -> list[mir.MOp]:
        """Lower a Reduce op to MThreadgroupReduce."""
        operand = self._resolve(op.operand)
        assert self.block_size is not None, "Reduce requires a block size (from arange)"
        num_sg = self.block_size // 32
        dtype = operand.type.dtype if hasattr(operand.type, "dtype") else "f32"

        shared_name = f"shared_reduce_{self._reduce_counter}"
        self._reduce_counter += 1

        result_ops = []
        if num_sg > 1:
            result_ops.append(
                mir.MThreadgroupAlloc(
                    alloc_name=shared_name, elem_type=_MSL_TYPES.get(dtype, "float"), size=num_sg
                )
            )

        reduce_op = mir.MThreadgroupReduce(
            reduce_op=op.op,
            operand=operand,
            shared_name=shared_name,
            block_size=self.block_size,
            sgid=self.sgid,
            slid=self.slid,
            dtype=dtype,
        )
        mv = mir.MValue(op.result.name, reduce_op.result_type(), reduce_op)
        reduce_op.result = mv
        self.value_map[op.result.name] = mv
        result_ops.append(reduce_op)
        return result_ops

    def _has_reduce(self) -> bool:
        """Check if the function uses Reduce (row-parallel pattern)."""

        def _check(ops):
            for op in ops:
                if isinstance(op, tir.Reduce):
                    return True
                if isinstance(op, tir.ForRange) and _check(op.body):
                    return True
            return False

        return _check(self.func.ops)

    def _has_shared(self) -> bool:
        """Check if the function uses SharedAlloc (needs threadgroup indexing)."""

        def _check(ops):
            for op in ops:
                if isinstance(op, (tir.SharedAlloc, tir.ThreadId)):
                    return True
                if isinstance(op, tir.ForRange) and _check(op.body):
                    return True
            return False

        return _check(self.func.ops)

    def _has_arange(self) -> bool:
        """Check if the function uses Arange (standard element-wise pattern)."""

        def _check(ops):
            for op in ops:
                if isinstance(op, tir.Arange):
                    return True
                if isinstance(op, tir.ForRange) and _check(op.body):
                    return True
            return False

        return _check(self.func.ops)

    def _set_grid(self):
        """Set grid and threadgroup dimensions based on the kernel pattern."""
        block = self.block_size or self.func.constexprs.get("BLOCK", 0)
        if block:
            self.mfunc.threadgroup_size = (block, 1, 1)
        else:
            # Row-wise kernel: one thread per program, no arange
            self.mfunc.threadgroup_size = (1, 1, 1)
        # Grid is set at dispatch time based on input size
        self.mfunc.grid = (0, 1, 1)  # 0 = determined at dispatch time
