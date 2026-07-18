import xml.etree.ElementTree as ET

from benchmarks.render_mlx_lm_results import _render_speedups, _render_throughput


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
                "model_config": {"num_hidden_layers": 16, "hidden_size": 2048},
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
                "model_config": {"num_hidden_layers": 24, "hidden_size": 896},
                "medians": {
                    "mlx_decode_tokens_per_second": 200.0,
                    "metile_decode_tokens_per_second": 198.0,
                    "decode_speedup": 0.99,
                    "end_to_end_speedup": 1.01,
                },
            },
        ]
    }


def test_throughput_report_is_accessible_svg():
    svg = _render_throughput(_suite())
    root = ET.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.attrib["role"] == "img"
    assert root.find("svg:title", namespace).text == "MLX-LM decode throughput across models"
    assert "Llama 3.2 1B" in svg
    assert "Qwen 2.5 0.5B" in svg
    assert "105.00" in svg


def test_speedup_report_contains_parity_and_both_metrics():
    svg = _render_speedups(_suite())
    root = ET.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:desc", namespace) is not None
    assert "native MLX = 1.000&#215;" in svg
    assert "decode 1.050&#215;" in svg
    assert "end-to-end 1.010&#215;" in svg
