"""The reports' charts (D-046): SVG files drawn by matplotlib, byte-identical on every rerun.

matplotlib is a dev dependency, imported only here: the reports need it, the service doesn't.
Colours follow the reference palette of the dataviz guidance, in light mode, on an opaque
surface, so a chart reads the same in GitHub's light and dark themes. Each chart sits above
a table with the same numbers, which is its accessible version.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
BEATEN = "#a3a19b"
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2: blue, orange
STYLE = {
    "svg.hashsalt": "support-rag",  # fixed element IDs
    "svg.fonttype": "none",  # text stays text: smaller, searchable
    "font.family": "sans-serif",
    "font.size": 9,
}
# Where a point's name goes, in order of preference: (dx, dy) in points, and alignment. The
# second ring is farther out, and a thin leader line joins the name to its point.
NEAR_SPOTS = (
    (7, 3, "left", "bottom"),
    (7, -3, "left", "top"),
    (-7, 3, "right", "bottom"),
    (-7, -3, "right", "top"),
    (0, 9, "center", "bottom"),
    (0, -9, "center", "top"),
)
FAR_SPOTS = (
    (14, 16, "left", "bottom"),
    (14, -16, "left", "top"),
    (-14, 16, "right", "bottom"),
    (-14, -16, "right", "top"),
    (0, 26, "center", "bottom"),
    (0, -26, "center", "top"),
    (22, 0, "left", "center"),
    (-22, 0, "right", "center"),
)


def _figure(width: float, height: float, rows: int = 1) -> tuple[Any, Any]:
    """A figure with `rows` panels stacked, each the full width, all on a 0-100% scale."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), facecolor=SURFACE)
    FigureCanvasAgg(figure)  # a renderer, to measure labels before saving
    axes = figure.subplots(rows, 1, squeeze=False)[:, 0]
    figure.subplots_adjust(hspace=0.35, top=0.93)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BEATEN)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0, pad=4)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% correct, with 95% CI", color=INK_SECONDARY)
    return figure, axes


def _save(figure: Any, path: Path) -> Path:
    """Write the SVG through a byte buffer: given a path, matplotlib writes text mode, which
    turns newlines into CRLF on Windows, so the file would differ by OS."""
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="svg",
        metadata={"Date": None},  # no timestamp, so a rerun gives the same file
        facecolor=SURFACE,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return path


def _overlap(box: Any, others: Sequence[Any]) -> float:
    """The area of `box` that other boxes cover, in square pixels."""
    total = 0.0
    for other in others:
        width = min(box.x1, other.x1) - max(box.x0, other.x0)
        height = min(box.y1, other.y1) - max(box.y0, other.y0)
        if width > 0 and height > 0:
            total += width * height
    return total


def _name_points(ax: Any, points: Sequence[tuple[float, float, str, tuple[float, float]]]) -> None:
    """Name each point (x, y, name, interval), in the order given, at the first spot that
    covers no other name, point or error bar and stays inside the plot; if there is none, at
    the spot that covers the least."""
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    taken = []
    for x, y, _, (low, high) in points:
        cx, cy = ax.transData.transform((x, y))
        _, bottom = ax.transData.transform((x, low))
        _, top = ax.transData.transform((x, high))
        taken.append(Bbox.from_extents(cx - 5, cy - 5, cx + 5, cy + 5))
        taken.append(Bbox.from_extents(cx - 1.5, bottom, cx + 1.5, top))
    for x, y, text, _ in points:
        best = None
        for spot in (*NEAR_SPOTS, *FAR_SPOTS):
            dx, dy, ha, va = spot
            probe = ax.annotate(
                text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=8,
            )  # fmt: skip
            box = probe.get_window_extent(renderer).expanded(1.05, 1.15)
            probe.remove()
            outside = not (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1))
            score = _overlap(box, taken) + (1e6 if outside else 0)
            if best is None or score < best[0]:
                best = (score, spot, box)
            if score == 0:
                break
        _, (dx, dy, ha, va), box = best
        leader = None
        if (dx, dy, ha, va) in FAR_SPOTS:
            leader = {"arrowstyle": "-", "color": BEATEN, "linewidth": 0.6, "shrinkA": 1,
                      "shrinkB": 5}  # fmt: skip
        ax.annotate(
            text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
            fontsize=8, color=INK, arrowprops=leader,
        )  # fmt: skip
        taken.append(box)


def answers_chart(result: Mapping[str, Any], path: Path) -> Path:
    """% correct against cost per answer (log scale) and against median latency, with each
    panel's frontier drawn as a line (D-046)."""
    import matplotlib
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    strategies = result["strategies"]
    with matplotlib.rc_context(STYLE):
        figure, (cost_ax, latency_ax) = _figure(8.5, 8.5, rows=2)
        panels = (
            (cost_ax, "cost", "cost_per_answer_usd", 1.0, "Generator cost per answer (log scale)"),
            (latency_ax, "latency", "latency_p50_ms", 1000.0, "Median latency (seconds)"),
        )
        for ax, measure, field, scale, xlabel in panels:
            frontier = result["frontier"][measure]["keys"]
            shown = {k: r for k, r in strategies.items() if r[field] is not None}
            if not shown:
                continue
            xs = [r[field] / scale for r in shown.values()]
            if measure == "cost":
                ax.set_xscale("log")
                ax.set_xlim(min(xs) / 1.8, max(xs) * 1.8)
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
            else:
                ax.set_xlim(0, max(xs) * 1.25)
            ax.set_xlabel(xlabel, color=INK_SECONDARY)
            line = [(strategies[k][field] / scale, strategies[k]["pct_correct"]) for k in frontier]
            if len(line) > 1:
                ax.plot(*zip(*line, strict=True), color=SERIES[0], linewidth=2, zorder=2)
            for key, row in shown.items():
                x, y = row[field] / scale, row["pct_correct"]
                low, high = row["ci"]
                colour = SERIES[0] if key in frontier else BEATEN
                face = SURFACE if key == "s_star" else colour
                edge = INK if key == "s_star" else SURFACE
                ax.errorbar(
                    [x], [y], yerr=[[y - low], [high - y]], fmt="none", ecolor=colour,
                    elinewidth=1, alpha=0.55, zorder=1,
                )  # fmt: skip
                ax.plot(
                    [x], [y], marker="o", markersize=7, markerfacecolor=face,
                    markeredgecolor=edge, markeredgewidth=1.5, linestyle="none", zorder=3,
                )  # fmt: skip
            first = sorted(shown, key=lambda k: k not in frontier)  # the frontier's names first
            _name_points(
                ax,
                [(shown[k][field] / scale, shown[k]["pct_correct"], shown[k]["short_name"],
                  shown[k]["ci"]) for k in first],
            )  # fmt: skip
        legend = [
            Line2D([], [], color=SERIES[0], linewidth=2, marker="o", markersize=7,
                   markeredgecolor=SURFACE, label="On the frontier"),
            Line2D([], [], color=BEATEN, linestyle="none", marker="o", markersize=7,
                   markeredgecolor=SURFACE, label="Beaten on both counts"),
            Line2D([], [], color=INK, linestyle="none", marker="o", markersize=7,
                   markerfacecolor=SURFACE, label="S*, gold articles (needs the answer key)"),
        ]  # fmt: skip
        figure.legend(
            handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3,
            frameon=False, fontsize=8, labelcolor=INK,
        )  # fmt: skip
        return _save(figure, path)


def rot_chart(result: Mapping[str, Any], path: Path) -> Path:
    """% correct against prompt tokens, one line per distractor type, both starting from the
    gold only (D-045)."""
    import matplotlib
    from matplotlib.transforms import offset_copy

    rows = {row["arm"]: row for row in result["rows"]}
    gold = rows[("gold only", 0)]
    with matplotlib.rc_context(STYLE):
        figure, (ax,) = _figure(6.0, 4.0)
        top = max(row["mean_prompt_tokens"] for row in rows.values()) / 1000
        ax.set_xlim(0, top * 1.15)
        ax.set_xlabel("Prompt tokens, mean per arm (thousands)", color=INK_SECONDARY)
        gold_xy = (gold["mean_prompt_tokens"] / 1000, gold["pct"])
        for colour, dodge, distractors in ((SERIES[0], -2.5, "hard"), (SERIES[1], 2.5, "random")):
            arms = sorted((a for a in rows if a[0] == distractors), key=lambda a: a[1])
            if not arms:
                continue
            shifted = offset_copy(ax.transData, fig=figure, x=dodge, y=0, units="points")
            xs = [gold_xy[0]] + [rows[a]["mean_prompt_tokens"] / 1000 for a in arms]
            ys = [gold_xy[1]] + [rows[a]["pct"] for a in arms]
            lows = [gold["ci"][0]] + [rows[a]["ci"][0] for a in arms]
            highs = [gold["ci"][1]] + [rows[a]["ci"][1] for a in arms]
            ax.errorbar(
                xs[1:], ys[1:], yerr=[[y - lo for y, lo in zip(ys[1:], lows[1:], strict=True)],
                                      [hi - y for y, hi in zip(ys[1:], highs[1:], strict=True)]],
                fmt="none", ecolor=colour, elinewidth=1, alpha=0.55, transform=shifted, zorder=1,
            )  # fmt: skip
            ax.plot(
                xs, ys, color=colour, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, transform=shifted, zorder=2,
                label=f"{distractors.capitalize()} distractors",
            )  # fmt: skip
            ax.annotate(
                distractors, (xs[-1], ys[-1]), xytext=(9 + dodge, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, color=INK,
            )  # fmt: skip
        ax.errorbar(
            [gold_xy[0]], [gold_xy[1]], yerr=[[gold_xy[1] - gold["ci"][0]],
                                              [gold["ci"][1] - gold_xy[1]]],
            fmt="none", ecolor=INK, elinewidth=1, alpha=0.55, zorder=3,
        )  # fmt: skip
        ax.plot(
            *gold_xy, marker="o", markersize=8, color=INK, markeredgecolor=SURFACE,
            markeredgewidth=1.5, linestyle="none", zorder=4, label="Gold only (S*)",
        )  # fmt: skip
        ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
        return _save(figure, path)
