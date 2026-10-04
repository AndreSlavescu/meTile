from pathlib import Path
from xml.etree import ElementTree

import pytest

from benchmarks.plots.render_diagrams import DIAGRAMS, Diagram, render

ROOT = Path(__file__).resolve().parents[2]
SVG = {"svg": "http://www.w3.org/2000/svg"}


@pytest.mark.parametrize("name", DIAGRAMS)
def test_published_diagram_matches_its_renderer(name):
    expected = DIAGRAMS[name]().serialize()
    assert (ROOT / "docs/_static" / f"{name}.svg").read_text() == expected


@pytest.mark.parametrize("name", DIAGRAMS)
def test_diagrams_have_accessible_descriptions_and_readable_text(name):
    root = ElementTree.fromstring(DIAGRAMS[name]().serialize())
    assert root.get("role") == "img"
    assert root.get("aria-labelledby") == "title description"
    assert root.find("svg:title", SVG).text
    assert root.find("svg:desc", SVG).text
    left, top, width, height = map(float, root.get("viewBox").split())
    assert (left, top, width) == (0, 0, 960)
    for label in root.findall("svg:text", SVG):
        size = float(label.get("font-size"))
        assert size >= 14
        assert 24 <= float(label.get("x")) <= width - 24
        assert size <= float(label.get("y")) <= height - 14


def test_render_creates_all_assets_in_the_requested_directory(tmp_path):
    destination = tmp_path / "diagrams"
    render(destination)
    assert {path.stem for path in destination.glob("*.svg")} == set(DIAGRAMS)
    for name, factory in DIAGRAMS.items():
        assert (destination / f"{name}.svg").read_text() == factory().serialize()


def test_card_rejects_clipped_body_text():
    diagram = Diagram("Title", "Subtitle", "Description", 300)
    with pytest.raises(ValueError, match="Text exceeds"):
        diagram.card(32, 124, 280, 60, "Stage", "A short card", ["This will not fit."])


def test_morton_and_linear_panels_keep_the_same_logical_tiles():
    root = ElementTree.fromstring(DIAGRAMS["morton-swizzle"]().serialize())
    orders = [int(cell.get("data-order")) for cell in root.findall("svg:rect[@data-order]", SVG)]
    assert orders[:16] == list(range(16))
    assert orders[16:] == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    assert sorted(orders[16:]) == orders[:16]


def test_tile_program_keeps_its_loop_indentation():
    root = ElementTree.fromstring(DIAGRAMS["tiling-overview"]().serialize())
    body = [label for label in root.findall("svg:text", SVG) if label.text.startswith("    ")]
    assert len(body) == 3
    assert all(
        label.get("{http://www.w3.org/XML/1998/namespace}space") == "preserve" for label in body
    )
