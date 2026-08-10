"""Draw the JAFFE ResNet-34 architecture — detailed, in an original flat style.

Deliberately NOT the isometric red-block look of common ResNet figures: flat
rounded blocks, a blue->teal depth gradient for the backbone and amber for the
head, with the information those figures usually omit — per-stage spatial size,
channel count, the number of residual blocks per stage (3-4-6-3, what makes this
ResNet-34 and not -18), and an inset expanding one BasicBlock with its skip
connection. Saved as PNG and SVG in jaffee/figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "figures")


def box(ax, x, y, w, h, color, title, sub=None, top=None, text_color="black",
        fontsize=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=color, edgecolor="#333333", lw=1.2,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=4, weight="bold")
    if sub:
        ax.text(x + w / 2, y - 0.22, sub, ha="center", va="top", fontsize=8.3,
                color="#333333", zorder=4)
    if top:
        ax.text(x + w / 2, y + h + 0.16, top, ha="center", va="bottom",
                fontsize=8.3, color="#555555", zorder=4)
    return x + w


def arrow(ax, x0, y0, x1, y1, color="#444444", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=12, color=color, lw=lw, zorder=2))


def draw():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(0.2, 8.9, "Architecture: ResNet-34 (ImageNet-pretrained) + JAFFE classification head",
            fontsize=13.5, weight="bold", ha="left", va="top")

    # ---- main pipeline (y ~ 6) ----
    y, h = 5.9, 1.5
    gap = 0.30
    x = 0.4

    x = box(ax, x, y, 1.05, h, "#e6e9f0", "Input", sub="224×224",
            top="3 ch") + gap
    arrow(ax, x - gap, y + h / 2, x, y + h / 2)

    x = box(ax, x, y, 1.25, h, "#4a6fa5", "Conv 7×7\n/2", sub="112×112",
            top="64", text_color="white") + gap
    arrow(ax, x - gap, y + h / 2, x, y + h / 2)

    x = box(ax, x, y, 1.15, h, "#8a97a8", "MaxPool\n3×3 /2", sub="56×56",
            top="64", text_color="white") + gap
    arrow(ax, x - gap, y + h / 2, x, y + h / 2)

    # 4 residual stages, teal deepening with depth.
    stages = [
        ("Stage 1", "3 blocks", "56×56", "64", "#7fd0c4"),
        ("Stage 2", "4 blocks", "28×28", "128", "#4db6ac"),
        ("Stage 3", "6 blocks", "14×14", "256", "#2a9d8f"),
        ("Stage 4", "3 blocks", "7×7", "512", "#1f7a70"),
    ]
    stage_centers = []
    for label, blocks, dim, ch, color in stages:
        title = f"{label}\n({blocks})"
        xe = box(ax, x, y, 1.5, h, color, title, sub=dim, top=ch,
                 text_color="white")
        stage_centers.append((x + 1.5 / 2, y + h))
        x = xe + gap
        arrow(ax, x - gap, y + h / 2, x, y + h / 2)

    # head
    x = box(ax, x, y, 1.2, h, "#b8c6d8", "Global\nAvgPool", sub="512") + gap
    arrow(ax, x - gap, y + h / 2, x, y + h / 2)
    x = box(ax, x, y, 1.0, h, "#f4c37d", "Dropout\n0.5") + gap
    arrow(ax, x - gap, y + h / 2, x, y + h / 2)
    x = box(ax, x, y, 1.15, h, "#f2a65a", "FC\n512→7") + gap
    arrow(ax, x - gap, y + h / 2, x, y + h / 2)
    x = box(ax, x, y, 1.15, h, "#83c76d", "Softmax", top="7 classes")

    # ImageNet-pretrained bracket over stem+stages, fine-tuned over head.
    bx0, bx1 = 1.75, stage_centers[-1][0] + 0.75
    by = y + h + 0.75
    ax.plot([bx0, bx0, bx1, bx1], [by - 0.12, by, by, by - 0.12], color="#4a6fa5",
            lw=1.4)
    ax.text((bx0 + bx1) / 2, by + 0.08, "backbone — ImageNet weights (fine-tuned)",
            ha="center", va="bottom", fontsize=10, color="#4a6fa5")

    # ---- inset: one BasicBlock detail ----
    ax.add_patch(FancyBboxPatch((0.4, 1.05), 15.4, 3.15,
                                boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="#f7f8fa", edgecolor="#aab2bd",
                                lw=1.2, ls="--", zorder=1))
    ax.text(0.65, 3.9, "Detail of one Residual Block (BasicBlock, ResNet-34) — "
            "each stage stacks several such blocks",
            fontsize=10.5, weight="bold", ha="left", va="center", color="#333333")

    iy, bh = 1.45, 1.0
    ax.text(0.72, iy + bh / 2, "x", fontsize=11, ha="center", va="center",
            style="italic")
    bx = 1.05
    for title, color, tc in [("Conv 3×3", "#4db6ac", "white"),
                             ("BN", "#9aa5b1", "white"),
                             ("ReLU", "#c7d0da", "black"),
                             ("Conv 3×3", "#4db6ac", "white"),
                             ("BN", "#9aa5b1", "white")]:
        nx = box(ax, bx, iy, 1.35, bh, color, title, text_color=tc, fontsize=9)
        arrow(ax, nx, iy + bh / 2, nx + 0.25, iy + bh / 2)
        bx = nx + 0.25

    sum_x = bx + 0.35
    ax.add_patch(plt.Circle((sum_x, iy + bh / 2), 0.28, facecolor="#f2a65a",
                            edgecolor="#333333", lw=1.2, zorder=3))
    ax.text(sum_x, iy + bh / 2, "+", fontsize=15, ha="center", va="center",
            zorder=4, weight="bold")
    arrow(ax, sum_x + 0.28, iy + bh / 2, sum_x + 0.9, iy + bh / 2)
    box(ax, sum_x + 0.9, iy, 1.2, bh, "#c7d0da", "ReLU", fontsize=9)
    ax.text(sum_x + 0.9 + 1.2 + 0.35, iy + bh / 2, "out", fontsize=11,
            ha="left", va="center", style="italic")

    # skip connection arc from input x, arcing above the boxes to the sum node
    ax.add_patch(FancyArrowPatch((0.72, iy + bh / 2 + 0.15), (sum_x, iy + bh + 0.02),
                                 connectionstyle="arc3,rad=-0.32",
                                 arrowstyle="-|>", mutation_scale=12,
                                 color="#e07a5f", lw=1.8, zorder=2))
    ax.text((0.72 + sum_x) / 2, iy + bh + 0.52, "skip connection (identity)",
            fontsize=8.8, ha="center", va="center", color="#e07a5f",
            style="italic")

    # per-stage block-count reminder (below the dashed inset)
    ax.text(0.4, 0.6, "ResNet-34 = stem + [3, 4, 6, 3] BasicBlocks + head   |   "
            "34 weighted layers total   |   JAFFE grayscale input duplicated to 3 channels",
            fontsize=9, ha="left", va="center", color="#555555")

    fig.savefig(os.path.join(OUT_DIR, "architecture.png"), dpi=150,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "architecture.svg"), bbox_inches="tight")
    plt.close(fig)
    print("ditulis: figures/architecture.png + .svg")


if __name__ == "__main__":
    draw()
