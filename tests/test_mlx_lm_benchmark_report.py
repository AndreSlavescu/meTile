import struct

import pytest

from benchmarks.render_mlx_lm_results import (
    _chart_data,
    _model_label,
    _render_speedups,
    _render_throughput,
)


def _suite():
    common = {
        "hardware": {"chip": "Apple M5", "memory": "32 GB"},
        "software": {"mlx": "0.32.0", "mlx_lm": "0.31.3"},
        "workload": {
            "prompt_tokens": 128,
            "generation_tokens": 256,
            "trials": 5,
            "seed": 0,
        },
    }
    return {
        "models": [
            {
                **common,
                "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "medians": {
                    "mlx_decode_tokens_per_second": 100.0,
                    "metile_decode_tokens_per_second": 105.0,
                    "decode_speedup": 1.05,
                    "end_to_end_speedup": 1.03,
                },
            },
            {
                **common,
                "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "medians": {
                    "mlx_decode_tokens_per_second": 200.0,
                    "metile_decode_tokens_per_second": 198.0,
                    "decode_speedup": 0.99,
                    "end_to_end_speedup": 1.01,
                },
            },
        ]
    }


def test_chart_data_preserves_absolute_and_relative_results():
    data = _chart_data(_suite())

    assert data["labels"] == ["Llama 3.2\n1B 4-bit", "Qwen 2.5\n0.5B 4-bit"]
    assert data["mlx"] == [100.0, 200.0]
    assert data["metile"] == [105.0, 198.0]
    assert data["decode_change"] == pytest.approx([5.0, -1.0])
    assert data["end_to_end_change"] == pytest.approx([3.0, 1.0])


def test_model_label_keeps_family_and_quantization_readable():
    assert _model_label("mlx-community/Qwen2.5-1.5B-Instruct-4bit") == ("Qwen 2.5\n1.5B 4-bit")


@pytest.mark.parametrize("renderer", (_render_throughput, _render_speedups))
def test_renderer_writes_high_resolution_png(renderer, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "chart.png"

    renderer(_suite(), output)

    payload = output.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", payload[16:24])
    assert width >= 1600
    assert height >= 900
    assert len(payload) > 20_000
