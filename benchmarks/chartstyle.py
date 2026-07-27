"""Shared chart style for the meTile benchmark renderers.

One simple form everywhere, one unit everywhere: speedup as a multiplier against
native MLX, with a parity rule at 1.0x. Thin marks, a single light gridline set,
no chart chrome beyond what carries meaning.

The palette is categorical slots 1-3 of the validated default data-viz palette.
Checked on a light surface with the palette validator (--pairs all): lightness band,
chroma floor, CVD separation and normal-vision floor all pass. Note the amber
(#ffb000) used by the older renderer fails the lightness band at 1.78:1 contrast,
which is why nothing here uses it.
"""

DECODE = "#2a78d6"
PREFILL = "#eb6834"
ACCENT = "#1baf7a"
FOURTH = "#eda100"

# Slots 1-3 clear the validator on the all-pairs list, so scatter and dot plots may use
# any of them together. The fourth slot only clears the adjacent list, so it is for line
# charts, where neighbouring series are what a reader compares.
SERIES = (DECODE, PREFILL, ACCENT, FOURTH)

INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#84837c"
GRID = "#e6e5e1"
RULE = "#a8a7a0"
SURFACE = "#ffffff"


def matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise ImportError(
            "Rendering benchmark charts requires the 'benchmarks' extra: "
            "pip install -e '.[benchmarks]'"
        ) from error
    return pyplot


def multiplier(value):
    """Format a speedup the same way in every chart."""
    return f"{value:.2f}x"


def frame(axis, grid_axis="x"):
    """Strip the chart down to one light gridline set and two soft spines."""
    axis.set_facecolor(SURFACE)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SOFT, labelsize=9.5, length=0)


_TOP_INCHES = 1.00
_BOTTOM_INCHES = 0.45


def headings(figure, title, subtitle, footer, left=0.045):
    """Place title/subtitle/footer at a fixed distance from the edge.

    Positioned in inches rather than figure fractions so the spacing holds whatever
    the figure height is - fractions collide as soon as a chart gets short.
    """
    height = figure.get_size_inches()[1]
    figure.text(
        left, 1 - 0.24 / height, title, fontsize=15, color=INK,
        fontweight="bold", ha="left", va="top",
    )
    if subtitle:
        figure.text(
            left, 1 - 0.60 / height, subtitle, fontsize=9.5, color=INK_SOFT,
            ha="left", va="top",
        )
    if footer:
        figure.text(
            0.985, 0.14 / height, footer, fontsize=7.5, color=INK_MUTED,
            ha="right", va="bottom",
        )


def layout_rect(figure):
    """Layout rectangle that reserves room for headings and footer."""
    height = figure.get_size_inches()[1]
    return (0.0, _BOTTOM_INCHES / height, 1.0, 1 - _TOP_INCHES / height)


def parity_rule(axis, orientation="vertical"):
    """Draw the 1.0x reference every chart is read against."""
    draw = axis.axvline if orientation == "vertical" else axis.axhline
    return draw(
        1.0,
        color=RULE,
        linewidth=1.3,
        linestyle=(0, (5, 4)),
        zorder=1,
        label="parity with MLX (1.00x)",
    )


def save(figure, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output, facecolor=SURFACE, metadata={"Software": "meTile benchmark renderer"}
    )
    print(f"Wrote {output}")
