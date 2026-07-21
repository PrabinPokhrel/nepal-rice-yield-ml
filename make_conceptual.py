"""Generate the conceptual trend-confounding schematic (Figure 1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

fig, ax = plt.subplots(figsize=(8.2, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

INK = "#1a1a1a"
NEUTRAL = "#e8eaf0"; NEUTRAL_E = "#8a93a6"
RED = "#f6d9d6"; RED_E = "#c0392b"
GREEN = "#d8eede"; GREEN_E = "#2e7d4f"

def box(x, y, w, h, text, fc, ec, bold=False, fs=9.5, tc=INK):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.6,rounding_size=2.2",
                 fc=fc, ec=ec, lw=1.6, mutation_scale=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", color=tc, zorder=5,
            linespacing=1.25)

def arrow(x1, y1, x2, y2, color=NEUTRAL_E, lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, lw=lw, color=color, shrinkA=2, shrinkB=2))

def label(x, y, text, color, fs=9):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            style="italic", color=color, weight="bold")

# --- shared top -------------------------------------------------------------
box(50, 93, 46, 9, "Observed rice yield\n(1963\u20132023)", NEUTRAL, NEUTRAL_E, bold=True, fs=10.5)
box(50, 78, 54, 10,
    "Intensification trend (fertilizer, HYV)  +  interannual climate variation",
    NEUTRAL, NEUTRAL_E, fs=9.5)
arrow(50, 88.5, 50, 83)

# split point
arrow(50, 73, 27, 65)
arrow(50, 73, 73, 65)
label(30, 69.5, "linear detrend", RED_E)
label(71, 69.5, "flexible detrend", GREEN_E)

# --- left (spurious) path ---------------------------------------------------
lx = 25
box(lx, 60, 40, 11,
    "Straight-line trend removed\n(a curved trend cannot be)", RED, RED_E, fs=9)
arrow(lx, 54.5, lx, 47)
box(lx, 41, 40, 12,
    "Residual = interannual climate\n+  leftover nonlinear trend", RED, RED_E, fs=9)
arrow(lx, 35, lx, 27.5)
box(lx, 21, 40, 12,
    "ML fits residual using\nco-trending climate variables\n(which proxy the leftover trend)",
    RED, RED_E, fs=8.8)
arrow(lx, 15, lx, 8.5)
box(lx, 3.5, 40, 8,
    "Apparent climate signal", RED, RED_E, bold=True, fs=10, tc=RED_E)
ax.text(lx, -3.2, "SPURIOUS  (trend explaining trend)", ha="center", va="center",
        fontsize=8.6, weight="bold", color=RED_E)

# --- right (honest) path ----------------------------------------------------
rx = 75
box(rx, 60, 40, 11,
    "Trend fully removed\n(spline or first-difference)", GREEN, GREEN_E, fs=9)
arrow(rx, 54.5, rx, 47)
box(rx, 41, 40, 12,
    "Residual = interannual\nclimate variation only", GREEN, GREEN_E, fs=9)
arrow(rx, 35, rx, 27.5)
box(rx, 21, 40, 12,
    "ML fits residual;\nno trend left for climate\nvariables to proxy",
    GREEN, GREEN_E, fs=8.8)
arrow(rx, 15, rx, 8.5)
box(rx, 3.5, 40, 8,
    "No robust climate signal", GREEN, GREEN_E, bold=True, fs=10, tc=GREEN_E)
ax.text(rx, -3.2, "the finding of this study", ha="center", va="center",
        fontsize=8.6, weight="bold", color=GREEN_E)

ax.set_ylim(-6, 100)
plt.tight_layout()
fig.savefig("figs/00_conceptual.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved figs/00_conceptual.png")
