"""Report charts rendered as inline SVG with matplotlib (headless)."""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display; must precede pyplot import

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
ACCENT = "#2563eb"
EMA_FAST = "#f59e0b"
EMA_SLOW = "#7c3aed"
BAND = "#93c5fd"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "investing-engine",  # deterministic SVG ids
    }
)


def _svg(figure: Figure) -> str:
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    svg = buffer.getvalue()
    return svg[svg.index("<svg") :]  # drop the XML prolog/doctype for inline embedding


def price_chart(series: pd.DataFrame, *, title: str) -> str:
    figure, axis = plt.subplots(figsize=(7.2, 2.6))
    index = series.index
    axis.fill_between(index, series["bb_low"], series["bb_high"], color=BAND, alpha=0.25, lw=0)
    axis.plot(index, series["Close"], color=ACCENT, lw=1.4, label="Close")
    axis.plot(index, series["ema34"], color=EMA_FAST, lw=1.0, label="EMA34")
    axis.plot(index, series["ema89"], color=EMA_SLOW, lw=1.0, label="EMA89")
    axis.set_title(title, pad=14)
    axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=3)
    figure.autofmt_xdate()
    return _svg(figure)


def rsi_chart(series: pd.DataFrame, *, predicted: float | None = None) -> str:
    figure, axis = plt.subplots(figsize=(7.2, 1.5))
    axis.plot(series.index, series["rsi"], color=INK, lw=1.0)
    axis.axhspan(70, 100, color="#fecaca", alpha=0.4, lw=0)
    axis.axhspan(0, 30, color="#bbf7d0", alpha=0.4, lw=0)
    if predicted is not None:
        axis.scatter([series.index[-1]], [predicted], color=ACCENT, zorder=3, s=14)
        axis.annotate(
            f"forecast {predicted:.1f}",
            (series.index[-1], predicted),
            textcoords="offset points",
            xytext=(-60, 6),
            color=ACCENT,
        )
    axis.set_ylim(0, 100)
    axis.set_yticks([30, 50, 70])
    axis.set_title("RSI (14)")
    figure.autofmt_xdate()
    return _svg(figure)


def history_chart(rows: Sequence[dict[str, Any]]) -> str | None:
    points = [
        (pd.Timestamp(r["timestamp"]), r["risk_score"])
        for r in rows
        if r.get("risk_score") is not None
    ]
    if len(points) < 2:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 1.4))
    when = pd.DatetimeIndex([p[0] for p in points])
    axis.plot(when, [p[1] for p in points], color=INK, lw=1.0, marker="o", ms=3)
    axis.set_ylim(0, 10)
    axis.set_yticks([0, 5, 10])
    axis.set_title("News risk across previous analyses")
    figure.autofmt_xdate()
    return _svg(figure)
