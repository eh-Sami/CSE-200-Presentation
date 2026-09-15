# New Part 3 figures:
#   eye_full / eye_zoom  - the supplied fundus photo, plus a marked zoom-in
#   fern_rules           - the fern coloured by WHICH of the 4 maps drew it
#   terrain_15           - a deep fractal surface (15 layers of noise)
import numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "images")
OUT = os.path.join(IMG, "part3")
RUST = "#9B4632"

# ---------------------------------------------------------------- #
# 1. Fundus photo: full frame with the zoom box marked, plus the crop
# ---------------------------------------------------------------- #
src = Image.open(os.path.join(IMG, "retinavessels.jpg")).convert("RGB")
BOX = (40, 250, 265, 470)                       # dense branching, lower left
full = src.copy()
d = ImageDraw.Draw(full)
d.rectangle(BOX, outline=RUST, width=6)
full.save(os.path.join(OUT, "eye_full.png"))

crop = src.crop(BOX)
crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
framed = Image.new("RGB", (crop.width + 16, crop.height + 16), RUST)
framed.paste(crop, (8, 8))
framed.save(os.path.join(OUT, "eye_zoom.png"))
print("eye_full.png / eye_zoom.png", full.size, framed.size)

# ---------------------------------------------------------------- #
# 2. The four rules, colour-coded: run the chaos game but remember
#    which map produced each point.
# ---------------------------------------------------------------- #
N = 160000
rng = np.random.default_rng(7)
r = rng.random(N)
xs = np.empty(N); ys = np.empty(N); wh = np.empty(N, dtype=np.int8)
x = y = 0.0
for i in range(N):
    p = r[i]
    if p < 0.01:
        x, y = 0.0, 0.16 * y; k = 0
    elif p < 0.86:
        x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.60; k = 1
    elif p < 0.93:
        x, y = 0.20 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.60; k = 2
    else:
        x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44; k = 3
    xs[i] = x; ys[i] = y; wh[i] = k

COLS = ["#C0392B", "#2E7D32", "#B8860B", "#1F6F8B"]
LABS = ["rule 1 - the stem",
        "rule 2 - the whole fern, shrunk and tilted",
        "rule 3 - the left leaflet",
        "rule 4 - the right leaflet"]
# No legend baked in -- the colour key lives on the slide as real text, so the
# figure stays tight and the wording stays editable.
fig, ax = plt.subplots(figsize=(3.4, 5.2))
SIZES = [2.0, 0.35, 0.45, 0.45]          # the stem is only 1% of the points
for k in range(4):
    m = wh == k
    ax.scatter(xs[m], ys[m], s=SIZES[k], c=COLS[k], marker=".", linewidths=0,
               alpha=0.9)
ax.set_aspect("equal"); ax.axis("off")
fig.patch.set_alpha(0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fern_rules.png"), dpi=200, transparent=True)
plt.close()
print("fern_rules.png")

# ---------------------------------------------------------------- #
# 3. Deep terrain: 15 layers of noise, each half the size of the last
# ---------------------------------------------------------------- #
ROWS, COLS = 840, 1480                  # wide, to fill the slide
rng = np.random.default_rng(11)

def octave(fx, fy):
    """One layer of smoothed value noise on a rectangular grid."""
    g = rng.normal(size=(fy + 1, fx + 1))
    ty = np.linspace(0, fy, ROWS, endpoint=False)
    tx = np.linspace(0, fx, COLS, endpoint=False)
    iy = ty.astype(int); wy = ty - iy; wy = wy * wy * (3 - 2 * wy)
    ix = tx.astype(int); wx = tx - ix; wx = wx * wx * (3 - 2 * wx)
    a = g[np.ix_(iy, ix)]; b = g[np.ix_(iy, ix + 1)]
    c = g[np.ix_(iy + 1, ix)]; e = g[np.ix_(iy + 1, ix + 1)]
    top = a * (1 - wx[None, :]) + b * wx[None, :]
    bot = c * (1 - wx[None, :]) + e * wx[None, :]
    return top * (1 - wy[:, None]) + bot * wy[:, None]

LAYERS, BASE = 15, 8
h = np.zeros((ROWS, COLS)); amp = 1.0
for k in range(LAYERS):
    fx = BASE * 2 ** k
    if fx >= COLS:                      # finer than a pixel: plain noise
        h += rng.normal(size=(ROWS, COLS)) * amp
    else:
        h += octave(fx, max(2, int(fx * ROWS / COLS))) * amp
    amp *= 0.55
h = (h - h.min()) / (h.max() - h.min())

terr = LinearSegmentedColormap.from_list(
    "t", ["#2f5f8a", "#4f8fb8", "#7fb069", "#a9c46c", "#c8b57a",
          "#8f6f4e", "#e8e8e8", "#ffffff"])
ls = LightSource(azdeg=315, altdeg=42)
# vert_exag scales with image width: the 256px panels used 8, so a 1340px
# field needs roughly 8*1340/256 to show the same relief.
fig, ax = plt.subplots(figsize=(COLS / 150.0, ROWS / 150.0))
ax.imshow(ls.shade(h, cmap=terr, vert_exag=350, blend_mode="soft"),
          interpolation="bilinear")
ax.axis("off")
ax.set_position([0, 0, 1, 1])
fig.patch.set_alpha(0)
plt.savefig(os.path.join(OUT, "terrain_15.png"), dpi=190, transparent=True,
            bbox_inches="tight", pad_inches=0)
plt.close()
print("terrain_15.png  (%d layers, %dx%d)" % (LAYERS, ROWS, COLS))
