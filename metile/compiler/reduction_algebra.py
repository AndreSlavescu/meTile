from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    """One expression in the reduction proof theory."""

    op: str
    args: tuple[Term, ...] = ()
    value: str | float | None = None

    def __str__(self):
        if self.op in {"symbol", "constant"}:
            return str(self.value)
        if self.op == "neg":
            return f"(-{self.args[0]})"
        if self.op == "exp":
            return f"exp({self.args[0]})"
        if self.op == "div":
            return f"({self.args[0]} / {self.args[1]})"
        separator = {"add": " + ", "mul": " * ", "max": ", "}[self.op]
        contents = separator.join(str(argument) for argument in self.args)
        return f"max({contents})" if self.op == "max" else f"({contents})"


State = tuple[Term, ...]
Merge = Callable[[State, State], State]
Lift = Callable[[Term, Term], State]
PairSummary = Callable[[Term, Term, Term, Term], State]
Finalize = Callable[[State], Term]


@dataclass(frozen=True)
class ReductionLaw:
    """A candidate list homomorphism represented by a finite summary state."""

    name: str
    components: tuple[str, ...]
    identity: State
    lift: Lift
    merge: Merge
    summarize_pair: PairSummary
    finalize: Finalize

    def __post_init__(self):
        if not self.components or len(self.identity) != len(self.components):
            raise ValueError("reduction components and identity must have equal nonzero arity")


@dataclass(frozen=True)
class ProofObligation:
    name: str
    left: State
    right: State
    verified: bool


@dataclass(frozen=True)
class ReductionCertificate:
    """Machine-checked obligations carried by an algorithmic rewrite."""

    theorem: str
    theory: str
    obligations: tuple[ProofObligation, ...]

    @property
    def verified(self) -> bool:
        return bool(self.obligations) and all(
            obligation.verified for obligation in self.obligations
        )


def symbol(name: str) -> Term:
    return Term("symbol", value=name)


def constant(value: float | str) -> Term:
    return Term("constant", value=value)


def add(*arguments: Term) -> Term:
    return Term("add", tuple(arguments))


def multiply(*arguments: Term) -> Term:
    return Term("mul", tuple(arguments))


def negate(argument: Term) -> Term:
    return Term("neg", (argument,))


def subtract(left: Term, right: Term) -> Term:
    return add(left, negate(right))


def exponential(argument: Term) -> Term:
    return Term("exp", (argument,))


def maximum(*arguments: Term) -> Term:
    return Term("max", tuple(arguments))


def normalize(term: Term) -> Term:
    """Normalize expressions in a restricted commutative-semiring/max/exp theory."""
    if term.op in {"symbol", "constant"}:
        return term
    arguments = tuple(normalize(argument) for argument in term.args)
    if term.op == "neg":
        argument = arguments[0]
        if _is_number(argument):
            return constant(-float(argument.value))
        if argument.op == "neg":
            return argument.args[0]
        return Term("neg", (argument,))
    if term.op == "max":
        flattened = _flatten("max", arguments)
        flattened = tuple(argument for argument in flattened if not _is_negative_infinity(argument))
        unique = tuple(sorted(set(flattened), key=_term_key))
        if not unique:
            return constant("-inf")
        return unique[0] if len(unique) == 1 else Term("max", unique)
    if term.op == "add":
        return _normalize_add(arguments)
    if term.op == "exp":
        argument = _normalize_add((arguments[0],))
        if _is_negative_infinity(argument):
            return constant(0.0)
        if _is_zero(argument):
            return constant(1.0)
        return Term("exp", (argument,))
    if term.op == "mul":
        return _normalize_multiply(arguments)
    if term.op == "div":
        return Term("div", arguments)
    raise ValueError(f"unsupported proof term: {term.op}")


def prove_reduction(law: ReductionLaw) -> ReductionCertificate:
    """Discharge identity, associativity, and list-homomorphism obligations."""
    score_a, score_b, score_c = (symbol(name) for name in ("a", "b", "c"))
    value_a, value_b, value_c = (symbol(name) for name in ("va", "vb", "vc"))
    state_a = law.lift(score_a, value_a)
    state_b = law.lift(score_b, value_b)
    state_c = law.lift(score_c, value_c)

    obligations = (
        _obligation("left_identity", law.merge(law.identity, state_a), state_a),
        _obligation("right_identity", law.merge(state_a, law.identity), state_a),
        _obligation(
            "generated_associativity",
            law.merge(law.merge(state_a, state_b), state_c),
            law.merge(state_a, law.merge(state_b, state_c)),
        ),
        _obligation(
            "pair_homomorphism",
            law.merge(state_a, state_b),
            law.summarize_pair(score_a, value_a, score_b, value_b),
        ),
    )
    return ReductionCertificate(
        theorem=law.name,
        theory="commutative-semiring + max-monoid + exp(x+y)=exp(x)exp(y)",
        obligations=obligations,
    )


def sum_reduction() -> ReductionLaw:
    return ReductionLaw(
        name="sum_monoid",
        components=("sum",),
        identity=(constant(0.0),),
        lift=lambda element, _: (element,),
        merge=lambda left, right: (add(left[0], right[0]),),
        summarize_pair=lambda left, _, right, __: (add(left, right),),
        finalize=lambda state: state[0],
    )


def max_reduction() -> ReductionLaw:
    return ReductionLaw(
        name="max_monoid",
        components=("maximum",),
        identity=(constant("-inf"),),
        lift=lambda element, _: (element,),
        merge=lambda left, right: (maximum(left[0], right[0]),),
        summarize_pair=lambda left, _, right, __: (maximum(left, right),),
        finalize=lambda state: state[0],
    )


def weighted_softmax_reduction() -> ReductionLaw:
    """Return the stable `(maximum, normalizer, numerator)` attention monoid."""

    def merge(left: State, right: State) -> State:
        merged_maximum = maximum(left[0], right[0])
        left_factor = exponential(subtract(left[0], merged_maximum))
        right_factor = exponential(subtract(right[0], merged_maximum))
        return (
            merged_maximum,
            add(multiply(left[1], left_factor), multiply(right[1], right_factor)),
            add(multiply(left[2], left_factor), multiply(right[2], right_factor)),
        )

    def summarize_pair(
        left_score: Term,
        left_value: Term,
        right_score: Term,
        right_value: Term,
    ) -> State:
        merged_maximum = maximum(left_score, right_score)
        left_weight = exponential(subtract(left_score, merged_maximum))
        right_weight = exponential(subtract(right_score, merged_maximum))
        return (
            merged_maximum,
            add(left_weight, right_weight),
            add(
                multiply(left_value, left_weight),
                multiply(right_value, right_weight),
            ),
        )

    return ReductionLaw(
        name="stable_weighted_softmax_monoid",
        components=("maximum", "normalizer", "numerator"),
        identity=(constant("-inf"), constant(0.0), constant(0.0)),
        lift=lambda score, value: (score, constant(1.0), value),
        merge=merge,
        summarize_pair=summarize_pair,
        finalize=lambda state: Term("div", (state[2], state[1])),
    )


def _obligation(name: str, left: State, right: State) -> ProofObligation:
    normalized_left = tuple(normalize(term) for term in left)
    normalized_right = tuple(normalize(term) for term in right)
    return ProofObligation(
        name, normalized_left, normalized_right, normalized_left == normalized_right
    )


def _normalize_add(arguments: tuple[Term, ...]) -> Term:
    flattened = list(_flatten("add", arguments))
    if any(_is_negative_infinity(argument) for argument in flattened):
        return constant("-inf")
    numeric = sum(float(argument.value) for argument in flattened if _is_number(argument))
    symbolic = [argument for argument in flattened if not _is_number(argument)]
    positive = []
    negative = []
    for argument in symbolic:
        if argument.op == "neg":
            negative.append(argument.args[0])
        else:
            positive.append(argument)
    for argument in tuple(positive):
        if argument in negative:
            positive.remove(argument)
            negative.remove(argument)
    terms = positive + [negate(argument) for argument in negative]
    if numeric:
        terms.append(constant(numeric))
    terms = sorted(terms, key=_term_key)
    if not terms:
        return constant(0.0)
    return terms[0] if len(terms) == 1 else Term("add", tuple(terms))


def _normalize_multiply(arguments: tuple[Term, ...]) -> Term:
    flattened = list(_flatten("mul", arguments))
    if any(_is_zero(argument) for argument in flattened):
        return constant(0.0)
    additive = next((argument for argument in flattened if argument.op == "add"), None)
    if additive is not None:
        remaining = list(flattened)
        remaining.remove(additive)
        return normalize(add(*(multiply(branch, *remaining) for branch in additive.args)))

    numeric = 1.0
    factors = []
    exponents = []
    for argument in flattened:
        if _is_number(argument):
            numeric *= float(argument.value)
        elif argument.op == "exp":
            exponents.append(argument.args[0])
        else:
            factors.append(argument)
    if numeric == 0.0:
        return constant(0.0)
    if exponents:
        factors.append(normalize(exponential(add(*exponents))))
    if numeric != 1.0 or not factors:
        factors.append(constant(numeric))
    factors = [factor for factor in factors if not _is_one(factor)]
    factors.sort(key=_term_key)
    if not factors:
        return constant(1.0)
    return factors[0] if len(factors) == 1 else Term("mul", tuple(factors))


def _flatten(operation: str, arguments: tuple[Term, ...]) -> tuple[Term, ...]:
    flattened = []
    for argument in arguments:
        if argument.op == operation:
            flattened.extend(argument.args)
        else:
            flattened.append(argument)
    return tuple(flattened)


def _term_key(term: Term):
    return term.op, str(term.value), tuple(_term_key(argument) for argument in term.args)


def _is_number(term: Term) -> bool:
    return term.op == "constant" and isinstance(term.value, (int, float))


def _is_zero(term: Term) -> bool:
    return _is_number(term) and float(term.value) == 0.0


def _is_one(term: Term) -> bool:
    return _is_number(term) and float(term.value) == 1.0


def _is_negative_infinity(term: Term) -> bool:
    return term.op == "constant" and term.value == "-inf"


__all__ = [
    "ProofObligation",
    "ReductionCertificate",
    "ReductionLaw",
    "Term",
    "max_reduction",
    "normalize",
    "prove_reduction",
    "sum_reduction",
    "weighted_softmax_reduction",
]
