"""Render the documentation's editable SVG diagrams without a graphics dependency.

Run ``python -m benchmarks.plots.render_diagrams`` to refresh all eight figures.
These are explanatory diagrams, not measurements or performance predictions.
"""

import argparse
import textwrap
from pathlib import Path
from xml.etree import ElementTree

INK = "#172b3a"
MUTED = "#536677"
BORDER = "#d8e3e9"
TEAL = "#087f8c"
ORANGE = "#c17134"
BLUE = "#3267a8"
PURPLE = "#7357aa"
SURFACE = "#f5f8fb"
TEAL_WASH = "#e8f5f4"
BLUE_WASH = "#eef3fb"
ORANGE_WASH = "#fcf2e8"
PURPLE_WASH = "#f3eff9"
WIDTH = 960


class Diagram:
    def __init__(self, title, subtitle, description, height):
        self.height = height
        self.root = ElementTree.Element(
            "svg",
            {
                "xmlns": "http://www.w3.org/2000/svg",
                "viewBox": f"0 0 {WIDTH} {height}",
                "role": "img",
                "aria-labelledby": "title description",
                "font-family": "Inter, DejaVu Sans, Arial, sans-serif",
            },
        )
        ElementTree.SubElement(self.root, "title", id="title").text = title
        ElementTree.SubElement(self.root, "desc", id="description").text = description
        definitions = ElementTree.SubElement(self.root, "defs")
        marker = ElementTree.SubElement(
            definitions,
            "marker",
            id="arrow",
            viewBox="0 0 10 10",
            refX="8",
            refY="5",
            markerWidth="6",
            markerHeight="6",
            orient="auto-start-reverse",
        )
        ElementTree.SubElement(
            marker,
            "path",
            d="M1 1L8 5L1 9",
            fill="none",
            stroke=MUTED,
            **{"stroke-width": "1.5", "stroke-linejoin": "round"},
        )
        self.rect(0, 0, WIDTH, height, fill="#ffffff", stroke="none", radius=0)
        self.text(32, 46, title, size=27, weight=650)
        self.text(32, 76, subtitle, color=MUTED)
        self.line(32, 100, 928, 100, color=BORDER)

    def rect(self, left, top, width, height, *, fill=SURFACE, stroke=BORDER, radius=12):
        return ElementTree.SubElement(
            self.root,
            "rect",
            x=str(left),
            y=str(top),
            width=str(width),
            height=str(height),
            rx=str(radius),
            fill=fill,
            stroke=stroke,
            **{"stroke-width": "1"},
        )

    def text(
        self, left, baseline, text, *, size=14, weight=400, color=INK, anchor="start", mono=False
    ):
        attributes = {
            "x": str(left),
            "y": str(baseline),
            "font-size": str(size),
            "font-weight": str(weight),
            "fill": color,
            "text-anchor": anchor,
        }
        if mono:
            attributes["font-family"] = "DejaVu Sans Mono, SFMono-Regular, Consolas, monospace"
            attributes["xml:space"] = "preserve"
        ElementTree.SubElement(self.root, "text", attributes).text = text

    def line(self, left, top, right, bottom, *, color=MUTED, arrow=False, both=False):
        attributes = {
            "x1": str(left),
            "y1": str(top),
            "x2": str(right),
            "y2": str(bottom),
            "stroke": color,
            "stroke-width": "1.5",
        }
        if arrow:
            attributes["marker-end"] = "url(#arrow)"
        if both:
            attributes["marker-start"] = "url(#arrow)"
        ElementTree.SubElement(self.root, "line", attributes)

    def route(self, path, *, arrow=True, dashed=False):
        attributes = {
            "d": path,
            "fill": "none",
            "stroke": MUTED,
            "stroke-width": "1.5",
            "stroke-linejoin": "round",
        }
        if arrow:
            attributes["marker-end"] = "url(#arrow)"
        if dashed:
            attributes["stroke-dasharray"] = "5 5"
        ElementTree.SubElement(self.root, "path", attributes)

    def card(self, left, top, width, height, eyebrow, title, lines, *, accent=TEAL, fill=SURFACE):
        self.rect(left, top, width, height, fill=fill)
        self.text(left + 20, top + 29, eyebrow.upper(), size=14, weight=650, color=accent)
        self.text(left + 20, top + 60, title, size=19, weight=650)
        baseline = top + 89
        for line in lines:
            for wrapped in textwrap.wrap(line, width=int((width - 40) / 7.6)):
                if baseline > top + height - 16:
                    raise ValueError(f"Text exceeds the {title!r} card")
                self.text(left + 20, baseline, wrapped, color=MUTED)
                baseline += 22

    def footer(self, first, second=None):
        top = self.height - (68 if second else 48)
        self.line(32, top, 928, top, color=BORDER)
        self.text(32, top + 27, first, color=MUTED)
        if second:
            self.text(32, top + 49, second, color=MUTED)

    def serialize(self):
        ElementTree.indent(self.root, space="  ")
        return ElementTree.tostring(self.root, encoding="unicode") + "\n"


def compilation_pipeline():
    diagram = Diagram(
        "From Python to an Apple GPU kernel",
        "Two entry points share the compiler; MLX adds its own validation and selection gates.",
        "A typed compute graph or a Python kernel enters planning and lowering. Graph discovery "
        "checks supported reduction laws; fusion uses exact min-cut for bipartite conflicts and "
        "greedy selection otherwise. Tile IR lowers to Metal IR, then MSL and Apple's compiler. "
        "Native dispatch and optional measured MLX selection are distinct runtime paths.",
        926,
    )
    diagram.card(
        32,
        124,
        432,
        134,
        "Graph frontend",
        "A typed compute graph",
        [
            "GraphBuilder or a supported integration.",
            "Shapes, dtypes, uses, and state stay explicit.",
        ],
        fill=BLUE_WASH,
        accent=BLUE,
    )
    diagram.card(
        496,
        124,
        432,
        134,
        "Kernel frontend",
        "A Python tile program",
        ["@metile.kernel with top-level tensor views.", "constexpr values specialize the program."],
        fill=TEAL_WASH,
    )
    diagram.line(248, 264, 248, 286, arrow=True)
    diagram.line(712, 264, 712, 286, arrow=True)
    diagram.card(
        32,
        294,
        432,
        174,
        "Match + fuse",
        "Choose legal graph regions",
        [
            "Recognize private attention subgraphs.",
            "Check the online-reduction certificate.",
            "Use min-cut for bipartite conflicts;",
            "use deterministic greedy selection otherwise.",
        ],
        accent=BLUE,
    )
    diagram.card(
        496,
        294,
        432,
        174,
        "Trace",
        "Build Tile IR",
        [
            "Record loads, stores, dot, and reductions.",
            "Keep loops, masks, layouts, and precision.",
            "Pointwise GEMM epilogues are expressions,",
            "not whole-kernel activation templates.",
        ],
    )
    diagram.route("M248 468V490H172V508")
    diagram.route("M712 468V490H172", arrow=False)
    diagram.card(
        32,
        516,
        280,
        168,
        "01 / Plan",
        "Plan the schedule",
        [
            "Use declared tile sizes.",
            "Choose backend and geometry.",
            "Check schedule requirements.",
        ],
        accent=BLUE,
    )
    diagram.card(
        340,
        516,
        280,
        168,
        "02 / Lower",
        "Build and optimize IR",
        ["Lower to Metal IR.", "Fold, vectorize, and stage.", "Make sync and MMA explicit."],
        accent=PURPLE,
    )
    diagram.card(
        648,
        516,
        280,
        168,
        "03 / Compile",
        "MSL → Metal toolchain",
        [
            "Emit MSL from Metal IR.",
            "Compile offline when supported;",
            "use a supported JIT fallback.",
        ],
        accent=ORANGE,
    )
    diagram.line(316, 594, 334, 594, arrow=True)
    diagram.line(624, 594, 642, 594, arrow=True)
    diagram.route("M788 684V706H248V728")
    diagram.route("M788 706H712V728")
    diagram.card(
        32,
        736,
        432,
        128,
        "Native runtime",
        "Dispatch the compiled pipeline",
        [
            "Bind buffers and check pipeline limits.",
            "Custom kernels need their own numerical tests.",
        ],
        accent=PURPLE,
    )
    diagram.card(
        496,
        736,
        432,
        128,
        "Optional MLX integration",
        "Validate, measure, then choose",
        [
            "Race valid candidates against native MLX.",
            "Keep the native path when a candidate fails.",
        ],
        fill=TEAL_WASH,
    )
    diagram.footer(
        "Apple's compiler emits device code. Neither a schedule nor a symbolic proof establishes a speedup."
    )
    return diagram


def algorithm_discovery():
    diagram = Diagram(
        "Recognize a graph. Check the reduction law.",
        "A certified rewrite is conditional; a failed match or obligation leaves the original graph intact.",
        "Attention recognition checks exact shapes, dtypes, axes and private intermediates. "
        "The weighted online-softmax summary is maximum, normalizer and numerator. A restricted "
        "symbolic checker verifies lifted identity, generated three-token associativity and "
        "two-token summary homomorphism. "
        "Passing produces a certificate and an atomic rewrite; failing preserves the original DAG. "
        "Symbolic real-arithmetic identities do not imply bitwise floating-point equivalence.",
        968,
    )
    diagram.card(
        32,
        126,
        432,
        294,
        "01 / Match",
        "Find an exact, private region",
        [
            "QKᵀ → scale → optional causal mask",
            "→ softmax over the key axis → PV",
            "",
            "Check shape, dtype, axis, and use counts.",
            "An intermediate with another user escapes",
            "the region and prevents this rewrite.",
        ],
        accent=BLUE,
        fill=BLUE_WASH,
    )
    diagram.card(
        496,
        126,
        432,
        294,
        "02 / Summarize",
        "Use a finite online state",
        [],
        accent=PURPLE,
        fill=PURPLE_WASH,
    )
    equations = (
        "Σ = (m, ℓ, o)     lift(s, v) = (s, 1, v)",
        "m = max(mₐ, mᵦ)",
        "α = exp(mₐ − m),  β = exp(mᵦ − m)",
        "ℓ = αℓₐ + βℓᵦ",
        "o = αoₐ + βoᵦ",
        "finalize(Σ) = o / ℓ",
    )
    for index, equation in enumerate(equations):
        diagram.text(516, 218 + index * 31, equation, size=15)
    diagram.line(468, 266, 490, 266, arrow=True)
    diagram.route("M712 420V446H480V466")
    diagram.rect(32, 474, 896, 226)
    diagram.text(52, 504, "03 / CHECK THE OBLIGATIONS", weight=650, color=TEAL)
    checks = (
        (52, "Lifted identity", ["identity ⊗ A = A", "A ⊗ identity = A"]),
        (344, "Generated associativity", ["(A ⊗ B) ⊗ C", "= A ⊗ (B ⊗ C)"]),
        (638, "Pair homomorphism", ["h([a, b])", "= A ⊗ B"]),
    )
    for left, heading, lines in checks:
        diagram.text(left, 549, heading, size=17, weight=650)
        for index, line in enumerate(lines):
            diagram.text(left, 585 + 27 * index, line, size=16)
    diagram.line(52, 630, 908, 630, color=BORDER)
    diagram.text(
        52,
        652,
        "A, B, C are lifted token states; h([a, b]) is the direct two-token summary.",
        color=MUTED,
    )
    diagram.text(
        52,
        679,
        "Restricted theory: commutative semiring, max monoid, and exp(x + y) = exp(x) exp(y).",
        color=MUTED,
    )
    diagram.route("M480 700V724H248V732", arrow=False)
    diagram.route("M480 724H712V732", arrow=False)
    diagram.text(248, 752, "All obligations pass", anchor="middle", color=TEAL)
    diagram.text(712, 752, "A match or obligation fails", anchor="middle", color=ORANGE)
    diagram.line(248, 758, 248, 766, arrow=True)
    diagram.line(712, 758, 712, 766, arrow=True)
    diagram.card(
        32,
        774,
        432,
        120,
        "Rewrite",
        "Attach the certificate",
        ["Replace the region atomically with flash_attention."],
        fill=TEAL_WASH,
    )
    diagram.card(
        496,
        774,
        432,
        120,
        "Keep",
        "Preserve the original DAG",
        ["Do not emit a speculative algorithm replacement."],
        accent=ORANGE,
        fill=ORANGE_WASH,
    )
    diagram.footer(
        "These checks concern symbolic real arithmetic. Validate floating-point results separately."
    )
    return diagram


def runtime_dispatch():
    diagram = Diagram(
        "Choose a measured winner, not a promised one",
        "Measured MLX primitives and model plans use these gates; not every graph region does.",
        "A cache lookup uses device, versions, source, shape, dtype and operation settings. "
        "On a miss, generated and native candidates pass numerical gates before interleaved "
        "timing. Family-specific switching margins preserve native MLX when no valid improvement "
        "is established. The winner is cached and dispatched through MLX or the native runtime.",
        960,
    )
    diagram.rect(32, 124, 896, 96, fill=BLUE_WASH)
    diagram.text(52, 154, "SELECTION KEY", weight=650, color=BLUE)
    diagram.text(
        52,
        181,
        "Device + architecture · framework / toolchain versions · source digest",
        color=MUTED,
    )
    diagram.text(
        52,
        203,
        "Shape + dtype · grid · operation settings such as scale / mask · candidate family",
        color=MUTED,
    )
    diagram.card(
        32,
        258,
        280,
        148,
        "Request",
        "An operator or plan",
        ["A measured primitive", "or a supported model plan."],
        accent=BLUE,
    )
    diagram.card(
        340,
        258,
        280,
        148,
        "Look up",
        "Persistent selection",
        ["Atomic disk + in-process cache.", "Require a matching signature."],
        accent=PURPLE,
    )
    diagram.card(
        648,
        258,
        280,
        148,
        "Cache hit",
        "Reuse the winner",
        ["A generated kernel or", "the original native operation."],
        fill=TEAL_WASH,
    )
    diagram.line(316, 332, 334, 332, arrow=True)
    diagram.line(624, 332, 642, 332, arrow=True)
    diagram.route("M480 406V438H172V480")
    diagram.text(190, 464, "Cache miss", color=MUTED)
    diagram.card(
        32,
        488,
        280,
        234,
        "Compare",
        "Keep native in the race",
        [
            "Native MLX primitive / graph.",
            "Generated meTile candidates:",
            "tiles, schedules, projections,",
            "and supported fusions.",
        ],
        accent=BLUE,
    )
    diagram.card(
        340,
        488,
        280,
        234,
        "Validate",
        "Check each candidate",
        [
            "Check shape, dtype, numerics.",
            "Check compile / launch status.",
            "Failed generated candidates",
            "keep the native path available.",
        ],
        accent=ORANGE,
    )
    diagram.card(
        648,
        488,
        280,
        234,
        "Measure",
        "Interleave and refine",
        [
            "Example model-plan search:",
            "3 provisional rounds;",
            "7 finalist pairs; rotate order.",
            "Family-specific margins.",
            "Audit model fidelity separately.",
        ],
        accent=PURPLE,
    )
    diagram.line(316, 604, 334, 604, arrow=True)
    diagram.line(624, 604, 642, 604, arrow=True)
    diagram.route("M788 722V746H480V778")
    diagram.route("M928 332H946V752H724V778", dashed=True)
    diagram.rect(32, 786, 896, 98, fill=TEAL_WASH)
    diagram.text(52, 818, "PERSIST + DISPATCH", weight=650, color=TEAL)
    diagram.text(
        52, 847, "MLX: native primitive or lazy custom Metal kernel sharing the existing arrays."
    )
    diagram.text(
        52,
        870,
        "Native runtime: prepared buffers / pipeline, shared encoders, batching, and repeats.",
    )
    diagram.footer(
        "A native fallback is not a generated-kernel speedup. Search settings and margins vary by family."
    )
    return diagram


def unified_memory():
    diagram = Diagram(
        "Shared memory, explicit synchronization",
        "On Apple silicon, the CPU and GPU can access the same Metal buffer allocation.",
        "Buffer(data=array) copies the input into shared Metal storage. CPU and GPU then "
        "access that allocation. After GPU writes, Buffer.numpy() waits for completion and "
        "returns a direct NumPy view. metile.shared is separate threadgroup-local storage.",
        652,
    )
    diagram.card(
        32,
        140,
        240,
        170,
        "CPU",
        "Python / NumPy",
        ["Create a Buffer.", "Read or write its view."],
        accent=BLUE,
    )
    diagram.card(
        328,
        140,
        304,
        170,
        "Shared allocation",
        "Metal buffer storage",
        ["CPU and GPU access", "the same physical allocation."],
        fill=TEAL_WASH,
    )
    diagram.card(
        688,
        140,
        240,
        170,
        "Apple GPU",
        "A Metal kernel",
        ["device pointers", "load / store"],
        accent=PURPLE,
    )
    diagram.line(282, 225, 318, 225, arrow=True, both=True)
    diagram.line(642, 225, 678, 225, arrow=True, both=True)
    diagram.text(
        480,
        346,
        "Buffer(data=array) copies input; later views share the buffer storage.",
        anchor="middle",
        color=MUTED,
    )
    for left, eyebrow, title, lines in (
        (32, "01 / Write", "Update the buffer", ["Metal stores the output."]),
        (340, "02 / Wait", "Wait for completion", ["numpy() synchronizes first."]),
        (648, "03 / View", "Read the shared result", ["result = output.numpy()"]),
    ):
        diagram.card(left, 402, 280, 142, eyebrow, title, lines)
    diagram.line(316, 475, 334, 475, arrow=True)
    diagram.line(624, 475, 642, 475, arrow=True)
    diagram.footer(
        "Keep the Buffer alive while using its NumPy view.",
        "metile.shared is threadgroup-local storage, not the host-visible shared allocation.",
    )
    return diagram


def _matrix(diagram, left, top, label, dimensions, selected, accent, wash):
    diagram.text(left, top, label, size=20, weight=650, color=accent)
    diagram.text(left + 240, top, dimensions, anchor="end", color=MUTED)
    for row in range(4):
        for column in range(4):
            active = (row, column) == selected
            diagram.rect(
                left + column * 60,
                top + 24 + row * 46,
                56,
                42,
                fill=accent if active else wash,
                stroke="none",
                radius=5,
            )
    diagram.text(left + 116, top + 236, label + " tile", anchor="middle", weight=650, color=accent)


def gemm_tiling():
    diagram = Diagram(
        "One output tile, many products along K",
        "C = A × B. Each program keeps an accumulator for its own BM × BN output tile.",
        "Matching BM by BK and BK by BN tiles are loaded along the reduction dimension. "
        "Each product updates the same FP32 accumulator. After the loop, a supported epilogue "
        "may run before the completed output tile is stored. Edge tiles need bounds checks.",
        802,
    )
    _matrix(diagram, 48, 156, "A", "M × K", (1, 0), TEAL, TEAL_WASH)
    _matrix(diagram, 360, 156, "B", "K × N", (0, 2), ORANGE, ORANGE_WASH)
    _matrix(diagram, 672, 156, "C", "M × N", (1, 2), PURPLE, PURPLE_WASH)
    diagram.text(321, 284, "×", size=26, anchor="middle", color=MUTED)
    diagram.text(633, 284, "→", size=26, anchor="middle", color=MUTED)
    for left, label in ((164, "BM × BK"), (476, "BK × BN"), (788, "BM × BN")):
        diagram.text(left, 421, label, anchor="middle", mono=True)
    diagram.text(
        480, 465, "Output grid = ceil(M / BM) × ceil(N / BN)", anchor="middle", color=MUTED
    )
    diagram.card(
        32,
        508,
        280,
        174,
        "01 / Initialize",
        "Start from zero",
        ["One FP32 accumulator tile", "belongs to this program."],
        accent=PURPLE,
    )
    diagram.card(
        340,
        508,
        280,
        174,
        "02 / Reduce",
        "Accumulate across K",
        [
            "Load the next A and B tiles.",
            "acc = dot(left, right, acc)",
            "Repeat in BK-sized steps.",
        ],
    )
    diagram.card(
        648,
        508,
        280,
        174,
        "03 / Store",
        "Write the result once",
        ["Apply a supported epilogue.", "Store the completed tile", "with output bounds checks."],
        accent=ORANGE,
    )
    diagram.line(316, 587, 334, 587, arrow=True)
    diagram.line(624, 587, 642, 587, arrow=True)
    diagram.footer(
        "Tile dimensions describe ownership, not one hardware instruction.",
        "The selected backend must support the requested shape, dtype, layout, and resource use.",
    )
    return diagram


def tiling_overview():
    diagram = Diagram(
        "Partition the output. Reuse the tile program.",
        "program_id chooses an output block; the reduction over K stays inside that program.",
        "A two-dimensional launch grid assigns one output tile to each program. The program "
        "initializes an accumulator, loads paired A and B tiles along K, accumulates dot "
        "products, and stores once. The compiler selects a supported matrix backend.",
        688,
    )
    diagram.card(
        32, 126, 368, 378, "01 / Partition C", "One program per output tile", [], accent=BLUE
    )
    for row in range(3):
        for column in range(3):
            active = row == column == 0
            left, top = 56 + column * 108, 218 + row * 70
            diagram.rect(
                left, top, 100, 62, fill=BLUE if active else BLUE_WASH, stroke="none", radius=6
            )
            label = f"pid ({row}, {column})" if row < 2 and column < 2 else "…"
            diagram.text(
                left + 50,
                top + 37,
                label,
                size=14,
                anchor="middle",
                color="#ffffff" if active else BLUE,
            )
    diagram.text(56, 465, "Tile shape: BM × BN", mono=True)
    diagram.text(56, 488, "Mask partial tiles at the edges.", color=MUTED)
    diagram.card(
        448, 126, 480, 378, "02 / Execute", "The same reduction for every tile", [], fill=TEAL_WASH
    )
    program = (
        "acc = zeros((BM, BN), dtype='f32')",
        "for offset in tile_range(0, K, BK):",
        "    left = A.load((row, offset))",
        "    right = B.load((offset, column))",
        "    acc = dot(left, right, acc)",
        "C.store((row, column), acc)",
    )
    for index, line in enumerate(program):
        diagram.text(470, 232 + index * 35, line, mono=True)
    diagram.line(406, 317, 440, 317, arrow=True)
    diagram.rect(32, 540, 896, 68, fill=PURPLE_WASH)
    diagram.text(
        52, 566, "THE COMPILER SELECTS A SUPPORTED HARDWARE LOWERING", color=PURPLE, weight=650
    )
    diagram.text(
        52,
        591,
        "SIMD-group matrix fragments · Metal 4 cooperative tensors · eligible NAX fragment schedules",
    )
    diagram.footer(
        "The logical ownership and K reduction stay the same; precision and layout limits still apply."
    )
    return diagram


def simdgroup_layout():
    diagram = Diagram(
        "Sixteen SIMD-groups share one output tile",
        "Example tensor-ops geometry: WM = 4, WN = 4, and a 128 × 128 program-owned tile.",
        "A four by four SIMD-group grid partitions a 128 by 128 output tile into sixteen "
        "32 by 32 subtiles. Each SIMD-group has 32 threads. The row and column indices are "
        "sgid divided by WN and sgid modulo WN. Hardware and pipeline limits still apply.",
        680,
    )
    diagram.text(335, 143, "OUTPUT COLUMNS", anchor="middle", color=MUTED, weight=650)
    for column in range(4):
        diagram.text(
            144 + column * 126,
            176,
            f"{column * 32}–{column * 32 + 31}",
            anchor="middle",
            color=MUTED,
        )
    for row in range(4):
        diagram.text(73, 235 + row * 78, f"{row * 32}–{row * 32 + 31}", anchor="end", color=MUTED)
        for column in range(4):
            left, top = 88 + column * 126, 197 + row * 78
            active = row == column == 0
            diagram.rect(
                left, top, 116, 68, fill=TEAL if active else TEAL_WASH, stroke="none", radius=7
            )
            color = "#ffffff" if active else TEAL
            diagram.text(
                left + 58,
                top + 29,
                f"sg({row},{column})",
                anchor="middle",
                size=15,
                weight=650,
                color=color,
            )
            diagram.text(left + 58, top + 52, "32 × 32", anchor="middle", color=color)
    diagram.card(
        638,
        148,
        290,
        364,
        "Hardware mapping",
        "32 threads per group",
        [
            "Each SIMD-group owns one",
            "32 × 32 output subtile.",
            "",
            "sg_row = sgid // WN",
            "sg_col = sgid % WN",
            "",
            "16 groups × 32 threads",
            "= 512 threads in this example.",
        ],
        accent=PURPLE,
        fill=PURPLE_WASH,
    )
    diagram.text(
        88,
        548,
        "Output rows are shown at left. The highlighted subtile belongs to sg(0,0).",
        color=MUTED,
    )
    diagram.footer(
        "This is one supported geometry, not a universal launch configuration.",
        "The planner and runtime must check device capabilities, fragment shapes, and pipeline thread limits.",
    )
    return diagram


def morton_swizzle():
    diagram = Diagram(
        "Change tile mapping, not the output",
        "Linear and Morton mappings cover the same 4 × 4 grid. Numbers show logical traversal indices.",
        "Linear row-major traversal uses rows 0,1,2,3; 4,5,6,7; 8,9,10,11; 12,13,14,15. "
        "The two-by-two Morton-panel mapping uses 0,1,4,5; 2,3,6,7; 8,9,12,13; 10,11,14,15. "
        "Local reuse may improve, but dispatch mapping does not guarantee physical GPU execution order.",
        706,
    )
    orders = (
        (32, "Linear row-major", tuple(range(16))),
        (496, "Morton: 2 × 2 panels", (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)),
    )
    for left, title, indices in orders:
        diagram.card(left, 124, 432, 388, "Logical traversal", title, [], accent=BLUE)
        for position, order in enumerate(indices):
            column, row = position % 4, position // 4
            cell_left, top = left + 64 + column * 76, 214 + row * 66
            early = order < 4
            cell = diagram.rect(
                cell_left, top, 68, 58, fill=BLUE if early else BLUE_WASH, stroke="none", radius=7
            )
            cell.set("data-order", str(order))
            diagram.text(
                cell_left + 34,
                top + 37,
                str(order),
                anchor="middle",
                size=21,
                weight=650,
                color="#ffffff" if early else BLUE,
            )
    diagram.rect(32, 542, 16, 16, fill=BLUE, stroke="none", radius=4)
    diagram.text(59, 555, "First four logical tiles", color=MUTED)
    diagram.text(
        32,
        591,
        "Morton groups nearby A-row and B-column regions, which may improve cache reuse.",
        color=MUTED,
    )
    diagram.footer(
        "A tile mapping does not guarantee the GPU's physical execution order.",
        "Measure candidate schedules; partial panels may use a valid fallback traversal.",
    )
    return diagram


DIAGRAMS = {
    "compilation-pipeline": compilation_pipeline,
    "algorithm-discovery": algorithm_discovery,
    "runtime-dispatch": runtime_dispatch,
    "unified-memory": unified_memory,
    "gemm-tiling": gemm_tiling,
    "tiling-overview": tiling_overview,
    "simdgroup-layout": simdgroup_layout,
    "morton-swizzle": morton_swizzle,
}


def render(output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, factory in DIAGRAMS.items():
        destination = output_directory / f"{name}.svg"
        destination.write_text(factory().serialize(), encoding="utf-8")
        print(f"Wrote {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/_static"))
    arguments = parser.parse_args()
    render(arguments.output_dir)


if __name__ == "__main__":
    main()
