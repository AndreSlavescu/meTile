import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.passes import fold_constants
from metile.ir import metal_ir as mir
from metile.ir.types import I32


@pytest.mark.parametrize(
    "operation,identity,parameter_on_left",
    [("add", 0, False), ("add", 0, True), ("sub", 0, True), ("mul", 1, False), ("mul", 1, True)],
)
def test_identity_folding_preserves_parameter_aliases_in_loop_bounds(
    operation, identity, parameter_on_left
):
    function = mir.MFunction(name="parameter_bound")
    function.params = [mir.MParam("size", I32, is_scalar=True)]
    parameter = mir.MValue("size", I32)
    constant = function.add_op(mir.MConstant(value=identity, dtype="i32"), "identity")
    operands = (parameter, constant) if parameter_on_left else (constant, parameter)
    bound = function.add_op(mir.MBinOp(op=operation, lhs=operands[0], rhs=operands[1]), "bound")
    function.ops.append(mir.MForLoop(iv_name="index", start=0, end=bound, step=1))

    fold_constants(function)
    source = emit(function)

    assert bound.defining_op in function.ops
    assert "int bound =" in source
    assert "index < bound" in source
