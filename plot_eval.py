"""Visualize AfriMGSM / PolyMath eval outputs from evaluate_math.py.

Usage:
    python plot_eval.py <eval.json> [<eval2.json> ...]

One file  -> headline metrics + per-problem correct-count histogram.
Many files -> pass@n / majority@n / average@n compared across checkpoints
              (files are ordered as passed on the command line).

Saves a PNG next to the first input file and also opens a window if possible.
"""

import json
import os
import sys

import matplotlib.pyplot as plt

# Okabe-Ito: a colorblind-safe categorical palette, fixed order.
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#E69F00", "#009E73", "#999999"
INK = "#222222"


def load(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "label": short_label(path),
        "step": step_of(path),
        "pass_at_n": d.get("pass_at_n_pct", 0.0),
        "majority": d.get("majority_vote_at_n_pct", 0.0),
        "average": d.get("average_at_n_pct", 0.0),
        "format_rate": d.get("format_rate", 0.0),
        "val_n": d.get("val_n"),
        "num_problems": d.get("num_problems"),
        "correct_counts": [r.get("num_correct", 0) for r in d.get("results", [])],
    }


def short_label(path):
    name = os.path.basename(path).replace(".json", "")
    # keep it readable: strip the common prefix noise, keep model/ckpt hints
    return name.replace("qwen3_", "").replace("_max1024", "").replace("_valn12", "")


def step_of(path):
    """Extract training step from filename: 'base' -> 0, 'stepN' -> N."""
    name = os.path.basename(path)
    if "base" in name:
        return 0
    import re

    m = re.search(r"step(\d+)", name)
    return int(m.group(1)) if m else None


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=INK, length=0)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_single(m, out_png):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- headline metrics (magnitude -> horizontal bars) ---
    metrics = [
        (f"pass@{m['val_n']}", m["pass_at_n"], BLUE),
        (f"majority@{m['val_n']}", m["majority"], ORANGE),
        (f"average@{m['val_n']}", m["average"], GREEN),
        ("format rate", m["format_rate"], GRAY),
    ]
    labels = [x[0] for x in metrics]
    vals = [x[1] for x in metrics]
    colors = [x[2] for x in metrics]
    y = range(len(metrics))
    ax1.barh(y, vals, color=colors, height=0.6)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(100, max(vals) * 1.15))
    ax1.set_xlabel("percent")
    for yi, v in zip(y, vals):
        ax1.text(v + 1, yi, f"{v:.1f}%", va="center", color=INK, fontsize=10)
    ax1.set_title(f"{m['label']}  (n={m['num_problems']} problems)", color=INK, fontsize=11)
    style_axes(ax1)

    # --- per-problem correct-count distribution ---
    n = m["val_n"] or (max(m["correct_counts"]) if m["correct_counts"] else 1)
    bins = list(range(n + 2))
    ax2.hist(m["correct_counts"], bins=bins, color=BLUE, rwidth=0.85, align="left")
    ax2.set_xlabel(f"# correct out of {n} samples")
    ax2.set_ylabel("# problems")
    ax2.set_xticks(range(n + 1))
    ax2.set_title("Per-problem correctness", color=INK, fontsize=11)
    style_axes(ax2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


def plot_compare(ms, out_png):
    # If every file has a parseable step, plot progress-over-steps as lines;
    # otherwise fall back to grouped bars ordered as passed.
    have_steps = all(m["step"] is not None for m in ms)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    # (label, key, color, linestyle). format rate is dashed -- it's a different
    # kind of metric (extractability), not an accuracy, but shares the % axis.
    series = [
        (f"pass@{ms[0]['val_n']}", "pass_at_n", BLUE, "-"),
        (f"majority@{ms[0]['val_n']}", "majority", ORANGE, "-"),
        (f"average@{ms[0]['val_n']}", "average", GREEN, "-"),
        ("format rate", "format_rate", GRAY, "--"),
    ]

    if have_steps:
        ms = sorted(ms, key=lambda m: m["step"])
        xs = [m["step"] for m in ms]
        for name, key, color, ls in series:
            ys = [m[key] for m in ms]
            ax.plot(xs, ys, ls, marker="o", color=color, linewidth=2, markersize=6, label=name)
        # direct-label the final point of each line
        for name, key, color, ls in series:
            ax.text(xs[-1] + 0.4, ms[-1][key], f"{ms[-1][key]:.1f}", color=INK, fontsize=9, va="center")
        ax.set_xlabel("training step")
        ax.set_xticks(xs)
    else:
        x = range(len(ms))
        w = 0.2
        for i, (name, key, color, ls) in enumerate(series):
            offs = [xi + (i - 1.5) * w for xi in x]
            ax.bar(offs, [m[key] for m in ms], width=w, color=color, label=name)
        ax.set_xticks(list(x))
        ax.set_xticklabels([m["label"] for m in ms], rotation=20, ha="right", fontsize=9)

    ax.set_ylabel("percent")
    ax.set_ylim(bottom=0)
    ax.set_title("Accuracy across checkpoints", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="best")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(1)
    ms = [load(p) for p in paths]
    for m in ms:
        print(
            f"{m['label']}: pass@{m['val_n']}={m['pass_at_n']:.1f}%  "
            f"majority={m['majority']:.1f}%  average={m['average']:.1f}%  "
            f"format={m['format_rate']:.1f}%"
        )
    out_png = os.path.splitext(paths[0])[0] + (
        "_summary.png" if len(ms) == 1 else "_compare.png"
    )
    if len(ms) == 1:
        plot_single(ms[0], out_png)
    else:
        plot_compare(ms, out_png)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
