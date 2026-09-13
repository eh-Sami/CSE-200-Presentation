# Figures for the two new Part 3 frames: Pythagorean tree + Barnsley fern.
# Both are generated from the construction rules, at three build stages each,
# so the slides can reveal them one step at a time.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
import os

PRIMARY = "#213448"; RAY = "#E69650"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "part3")

# ---------------------------------------------------------------- #
# Pythagorean tree: one square, then two smaller squares on the legs
# of a right triangle built on its top edge.  Points as complex nums.
# ---------------------------------------------------------------- #
def squares(depth, alpha=np.pi / 4):
    out = []
    def rec(a, b, d):
        v = b - a
        n = 1j * v                       # perpendicular, square sits to the left
        out.append((d, [a, b, b + n, a + n]))
        if d >= depth:
            return
        A, B = a + n, b + n              # top edge of this square
        P = A + np.cos(alpha) * v * np.exp(1j * alpha)   # right-angle apex
        rec(A, P, d + 1)
        rec(P, B, d + 1)
    rec(0 + 0j, 1 + 0j, 0)
    return out

# Colour by recursion level: bark at the trunk, leaf green at the tips.
# Thin light gaps between squares keep the deep levels as fine filigree
# rather than one solid blob.
bark = LinearSegmentedColormap.from_list(
    "bark", ["#5a3a1e", "#7a5a2b", "#4f7a35", "#6fb043", "#a8dc72"])

def extent(items, pad=0.15):
    pts = np.array([p for _, poly in items for p in poly])
    return ((pts.real.min() - pad, pts.real.max() + pad),
            (pts.imag.min() - pad, pts.imag.max() + pad))

def draw(items, name, label, xlim, ylim, maxdepth, lw=0.25):
    # Shape the canvas to the tree so nothing is wasted on letterboxing; the
    # 1.15 leaves room for the title. The growth panels pass a shared extent,
    # so they stay in a common frame and the tree grows in place.
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    fw = 4.4
    fig, ax = plt.subplots(figsize=(fw, fw * dy / dx * 1.15))
    ax.add_collection(PolyCollection(
        [[(p.real, p.imag) for p in poly] for _, poly in items],
        facecolors=[bark(d / max(maxdepth, 1)) for d, _ in items],
        edgecolors="white", linewidths=lw))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(label, fontsize=15, color=PRIMARY, fontweight="bold")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=200, transparent=True)
    plt.close()
    print("%s  (%d squares)" % (name, len(items)))

DEPTHS = [3, 7, 12]
full = squares(max(DEPTHS))
xlim, ylim = extent(full)
for k, depth in enumerate(DEPTHS, start=1):
    sq = [(d, poly) for d, poly in full if d <= depth]
    draw(sq, "pythagoras_%d.png" % k, "%d levels" % depth, xlim, ylim, depth)

# Tilted rule: alpha != 45 deg makes the two children different sizes
# (cos a vs sin a), so the tree leans instead of mirroring itself.
TILT, LEAN_DEPTH = 55, 16
lean = squares(LEAN_DEPTH, alpha=np.deg2rad(TILT))
lx, ly = extent(lean)
draw(lean, "pythagoras_4.png", "tilted rule", lx, ly, LEAN_DEPTH, lw=0.12)

# ---------------------------------------------------------------- #
# Barnsley fern: 4 affine maps, chosen at random with fixed weights.
# Same point sequence drawn at 3 densities so it emerges from noise.
# ---------------------------------------------------------------- #
N = 200000
rng = np.random.default_rng(7)
r = rng.random(N)
xs = np.empty(N); ys = np.empty(N)
x = y = 0.0
for i in range(N):
    p = r[i]
    if p < 0.01:
        x, y = 0.0, 0.16 * y
    elif p < 0.86:
        x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.60
    elif p < 0.93:
        x, y = 0.20 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.60
    else:
        x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44
    xs[i] = x; ys[i] = y

fxlim = (xs.min() - 0.3, xs.max() + 0.3)
fylim = (ys.min() - 0.3, ys.max() + 0.3)

for k, n in enumerate([2000, 25000, N], start=1):
    fig, ax = plt.subplots(figsize=(3.1, 4.6))
    ax.scatter(xs[:n], ys[:n], s=0.35, c="#2e7d32", marker=".", linewidths=0, alpha=0.8)
    ax.set_xlim(*fxlim); ax.set_ylim(*fylim)
    ax.set_aspect("equal"); ax.axis("off")
    lab = "%s dots" % ("{:,}".format(n))
    ax.set_title(lab, fontsize=14, color=PRIMARY, fontweight="bold")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    f = os.path.join(OUT, "fern_%d.png" % k)
    plt.savefig(f, dpi=200, transparent=True)
    plt.close()
    print("fern_%d.png  (%d dots, %.0f KB)" % (k, n, os.path.getsize(f) / 1024))
