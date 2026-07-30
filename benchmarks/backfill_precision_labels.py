"""Attach a precision label to published results that were written before labels existed.

Three artifacts under `benchmarks/results` report end-to-end model speedups with no statement of the
precision they were measured at, because they were written at schema 2, 3 and 5 and the suite only
began recording it at 19. They are the most quotable numbers in the repository and the only ones
without a label, which is how `m5-mlx-lm-bf16-models.json` came to show decode speedups of 1.37x to
1.75x that are a representation change rather than a kernel win: meTile quantizes the down projection
to affine8 there while MLX runs bf16.

Re-running the suites would be the ideal fix and is not required, because the label is a function of
which features ran and every one of these files records that in `selected_plan`. The label is
therefore derived, not invented, and it is derived by calling the same `_precision_comparison` the
writer uses rather than by reimplementing its rules — a second implementation would be free to drift
from the first, and the audit that checks labels against plans would then be checking a copy.

Each label written here is marked `derived_from: selected_plan` so nobody mistakes it for something
the original run emitted, and the baseline dtype is left as the function reports it: those schemas did
not record a model dtype, and inferring one from the filename would be a guess dressed as data.

usage:
    python benchmarks/backfill_precision_labels.py --check   # report, change nothing
    python benchmarks/backfill_precision_labels.py           # write the labels
"""

import argparse
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

from benchmarks.mlx_lm_backend import _precision_comparison

RESULTS = Path(__file__).resolve().parent / "results"

# Every feature `_precision_comparison` consults. Older plans omit the ones that did not exist yet,
# and it indexes rather than gets, so the plan has to be completed before it is passed in. Defaulting
# a missing feature to False is the truth: a feature absent from the schema did not run.
PLAN_FEATURES = (
    "affine_prefill",
    "attention",
    "compressed_attention",
    "compressed_down",
    "compressed_gate_up",
    "compressed_vocab",
    "dense_mlp",
    "dense_residual",
    "graph_fusion",
    "quantized_mlp",
    "rms_norm",
)


class _Plan:
    """Adapts a recorded plan dict to the interface `_precision_comparison` expects."""

    def __init__(self, recorded):
        self._features = {feature: bool(recorded.get(feature)) for feature in PLAN_FEATURES}

    def as_dict(self):
        return dict(self._features)


class _Weights:
    """Stands in for a prepared compressed-weight object, carrying only its format."""

    def __init__(self, weight_format):
        self.format = weight_format


def _entries(document):
    models = document.get("models")
    if isinstance(models, list):
        return [m for m in models if isinstance(m, dict)]
    if isinstance(models, dict):
        return [m for m in models.values() if isinstance(m, dict)]
    return []


def _label_for(entry):
    """The precision label this record's own plan implies, or None if it cannot be derived."""
    recorded = entry.get("selected_plan") or entry.get("candidate_plan")
    if not isinstance(recorded, dict):
        return None
    plan = _Plan(recorded)
    weights = None
    if plan.as_dict()["compressed_down"]:
        weight_format = (entry.get("compressed_down") or {}).get("format")
        if not weight_format:
            return None
        weights = _Weights(weight_format)
    label = _precision_comparison(plan, weights, entry.get("model_config"))
    label["derived_from"] = "selected_plan"
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report without writing")
    arguments = parser.parse_args()

    missing = 0
    for path in sorted(RESULTS.glob("*.json")):
        document = json.loads(path.read_text())
        entries = _entries(document)
        if document.get("precision_comparison") or not entries:
            continue
        if all(isinstance(entry.get("precision_comparison"), dict) for entry in entries):
            continue

        labelled = []
        for entry in entries:
            if isinstance(entry.get("precision_comparison"), dict):
                labelled.append(entry["precision_comparison"]["class"])
                continue
            label = _label_for(entry)
            if label is None:
                print(f"{path.name}: cannot derive a label for {entry.get('model', '?')}")
                missing += 1
                continue
            entry["precision_comparison"] = label
            labelled.append(label["class"])

        classes = sorted(set(labelled))
        print(f"{path.name}: {len(labelled)} records -> {', '.join(classes)}")
        if not arguments.check:
            path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    if arguments.check:
        print("\n--check: nothing written")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
