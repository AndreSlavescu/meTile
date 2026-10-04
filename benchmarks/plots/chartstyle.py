"""Shared typography, reference rules, and vector/raster benchmark exports."""

DECODE = "#2166ac"
PREFILL = "#b44b24"
ACCENT = "#19755b"
FOURTH = "#7854a4"

SERIES = (DECODE, PREFILL, ACCENT, FOURTH)

INK = "#172536"
INK_SOFT = "#465365"
INK_MUTED = "#5c6978"
GRID = "#e4e9ef"
RULE = "#748396"
SURFACE = "#ffffff"


def matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        pyplot.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
                "font.size": 10,
                "axes.labelcolor": INK_SOFT,
                "axes.titlecolor": INK,
                "svg.fonttype": "none",
                "svg.hashsalt": "metile-benchmarks",
                "savefig.dpi": 180,
            }
        )
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
        left,
        1 - 0.24 / height,
        title,
        fontsize=15,
        color=INK,
        fontweight="bold",
        ha="left",
        va="top",
    )
    if subtitle:
        figure.text(
            left,
            1 - 0.60 / height,
            subtitle,
            fontsize=9.5,
            color=INK_SOFT,
            ha="left",
            va="top",
        )
    if footer:
        figure.text(
            0.985,
            0.14 / height,
            footer,
            fontsize=8,
            color=INK_MUTED,
            ha="right",
            va="bottom",
        )


def layout_rect(figure):
    """Layout rectangle that reserves room for headings and footer."""
    height = figure.get_size_inches()[1]
    return (0.0, _BOTTOM_INCHES / height, 1.0, 1 - _TOP_INCHES / height)


def parity_rule(axis, orientation="vertical", reference="MLX"):
    """Draw the 1.0x reference every chart is read against."""
    draw = axis.axvline if orientation == "vertical" else axis.axhline
    return draw(
        1.0,
        color=RULE,
        linewidth=1.3,
        linestyle=(0, (5, 4)),
        zorder=1,
        label=f"parity with {reference} (1.00x)",
    )


def save(figure, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".png"):
        destination = output.with_suffix(suffix)
        metadata = (
            {"Creator": "meTile benchmark renderer", "Date": None}
            if suffix == ".svg"
            else {"Software": "meTile benchmark renderer"}
        )
        figure.savefig(destination, facecolor=SURFACE, metadata=metadata)
        print(f"Wrote {destination}")
