"""Shared, document-width typography and honest vector/raster benchmark exports."""

from itertools import combinations
from textwrap import fill

DECODE = "#3267a8"
PREFILL = "#b4642c"
ACCENT = "#087f8c"
FOURTH = "#7357aa"

SERIES = (DECODE, PREFILL, ACCENT, FOURTH)

INK = "#172b3a"
INK_SOFT = "#40596b"
INK_MUTED = "#536677"
GRID = "#e3e9ed"
RULE = "#82919c"
SURFACE = "#ffffff"
ROW_SURFACE = "#f3f6f8"
WIDTH = 9.0
DPI = 200


def matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        pyplot.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
                "font.size": 11,
                "axes.labelcolor": INK_SOFT,
                "axes.titlecolor": INK,
                "axes.labelpad": 10,
                "axes.titlepad": 14,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": False,
                "legend.fontsize": 10.5,
                "legend.frameon": False,
                "legend.labelcolor": INK_SOFT,
                "lines.solid_capstyle": "round",
                "svg.fonttype": "none",
                "svg.hashsalt": "metile-benchmarks",
                "savefig.dpi": DPI,
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
    """Keep one quiet grid, an unobtrusive baseline, and readable ticks."""
    axis.set_facecolor(SURFACE)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SOFT, labelsize=10.5, length=0, pad=8)


_TOP_INCHES = 1.25
_BOTTOM_INCHES = 0.65


def wrapped(text, width=98):
    """Wrap prose without merging explicit lines or breaking identifiers."""
    return "\n".join(
        fill(line, width, break_long_words=False, break_on_hyphens=False)
        for line in text.splitlines()
    )


def headings(figure, title, subtitle, footer, left=0.045):
    """Place title/subtitle/footer at a fixed distance from the edge.

    Positioned in inches rather than figure fractions so the spacing holds whatever
    the figure height is - fractions collide as soon as a chart gets short.
    """
    height = figure.get_size_inches()[1]
    figure.text(
        left,
        1 - 0.22 / height,
        title,
        fontsize=17,
        color=INK,
        fontweight="bold",
        ha="left",
        va="top",
    )
    if subtitle:
        figure.text(
            left,
            1 - 0.62 / height,
            wrapped(subtitle),
            fontsize=10.5,
            color=INK_SOFT,
            ha="left",
            va="top",
        )
    if footer:
        figure.text(
            left,
            0.15 / height,
            wrapped(footer, 108),
            fontsize=9,
            color=INK_MUTED,
            ha="left",
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
        linewidth=1.2,
        linestyle=(0, (4, 4)),
        zorder=1,
        label=f"parity with {reference} (1.00x)",
    )


def comparison_rows(axis, labels, series, positions=None, columns=(0.85, 0.97), digits=2):
    """Pair uncluttered dot positions with aligned, explicitly labeled value columns.

    ``series`` contains (values, color, name) triples. Every value is plotted;
    rounding is restricted to the printed labels. Columns use figure coordinates
    so long model labels and near-parity measurements never collide with values.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    if len(series) != len(columns):
        raise ValueError("every measurement series needs its own value column")
    positions = list(range(len(labels))) if positions is None else positions
    transform = blended_transform_factory(axis.figure.transFigure, axis.transData)
    for index, position in enumerate(positions):
        if index % 2 == 0:
            axis.add_patch(
                Rectangle(
                    (0.035, position - 0.46),
                    0.945,
                    0.92,
                    transform=transform,
                    facecolor=ROW_SURFACE,
                    edgecolor="none",
                    clip_on=False,
                    zorder=-2,
                )
            )
        axis.text(
            0.045,
            position,
            labels[index],
            transform=transform,
            va="center",
            fontsize=10.5,
            color=INK,
            clip_on=False,
        )
    for index, ((values, color, name), column) in enumerate(zip(series, columns, strict=True)):
        offset = (index - (len(series) - 1) / 2) * 0.28
        axis.scatter(
            values,
            [position + offset for position in positions],
            s=45,
            marker=("o", "s", "D", "^")[index % 4],
            color=color,
            edgecolors=SURFACE,
            linewidths=0.9,
            zorder=3,
            label=name,
        )
        for position, value in zip(positions, values, strict=True):
            axis.text(
                column,
                position,
                f"{value:.{digits}f}x",
                transform=transform,
                ha="right",
                va="center",
                fontsize=10.5,
                color=color,
                clip_on=False,
            )
    axis.set_yticks([])
    frame(axis)


def value_headers(axis, series, columns=(0.85, 0.97), label="MODEL / FORMAT"):
    """Label the model and value columns without a legend covering measurements."""
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(axis.figure.transFigure, axis.transAxes)
    axis.text(0.045, 1.025, label, transform=transform, fontsize=9, color=INK_MUTED)
    for (_, color, name), column in zip(series, columns, strict=True):
        axis.text(
            column,
            1.025,
            name,
            transform=transform,
            fontsize=10,
            color=color,
            fontweight="bold",
            ha="right",
            va="bottom",
        )


def validate_text_layout(figure):
    """Reject clipped or colliding labels, excluding ticks outside the view limits."""
    from matplotlib.text import Text

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    offscreen = set()
    for axis in figure.axes:
        for dimension in (axis.xaxis, axis.yaxis):
            lower, upper = sorted(dimension.get_view_interval())
            for tick in (*dimension.get_major_ticks(), *dimension.get_minor_ticks()):
                if not lower <= tick.get_loc() <= upper:
                    offscreen.update((tick.label1, tick.label2))
    texts = [
        artist
        for artist in figure.findobj(Text)
        if artist.get_visible() and artist.get_text().strip() and artist not in offscreen
    ]
    bounds = [
        (
            artist,
            artist.get_window_extent(renderer).padded(6 if artist.get_gid() == "bar-value" else 2),
        )
        for artist in texts
    ]
    canvas = figure.bbox
    for artist, box in bounds:
        if box.x0 < canvas.x0 or box.y0 < canvas.y0 or box.x1 > canvas.x1 or box.y1 > canvas.y1:
            raise RuntimeError(f"chart text leaves the canvas: {artist.get_text()!r}")
    for (left, left_box), (right, right_box) in combinations(bounds, 2):
        if left_box.overlaps(right_box):
            raise RuntimeError(f"chart text overlaps: {left.get_text()!r} and {right.get_text()!r}")


def save(figure, output):
    validate_text_layout(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".png"):
        destination = output.with_suffix(suffix)
        metadata = (
            {"Creator": "meTile benchmark renderer", "Date": None}
            if suffix == ".svg"
            else {"Software": "meTile benchmark renderer"}
        )
        figure.savefig(destination, facecolor=SURFACE, metadata=metadata)
        if suffix == ".svg":
            destination.write_text(
                "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
            )
        print(f"Wrote {destination}")
