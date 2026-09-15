# Voxel world: the SAME fractal noise as terrain_15, but quantised into whole
# blocks and drawn isometrically. That is how block-based sandbox games build
# their landscapes -- noise in, blocks out. Original render, original palette.
import numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "part3")

N = 64                      # columns per side
LEVELS = 16                 # how many block heights
LAYERS, BASE = 15, 2
rng = np.random.default_rng(5)

def octave(f):
    g = rng.normal(size=(f + 1, f + 1))
    t = np.linspace(0, f, N, endpoint=False)
    i = t.astype(int); w = t - i; w = w * w * (3 - 2 * w)
    a = g[np.ix_(i, i)]; b = g[np.ix_(i + 1, i)]
    c = g[np.ix_(i, i + 1)]; d = g[np.ix_(i + 1, i + 1)]
    return ((a * (1 - w[:, None]) + b * w[:, None]) * (1 - w[None, :]) +
            (c * (1 - w[:, None]) + d * w[None, :]) * w[None, :])

h = np.zeros((N, N)); amp = 1.0
for k in range(LAYERS):
    f = BASE * 2 ** k
    h += (rng.normal(size=(N, N)) if f >= N else octave(f)) * amp
    amp *= 0.55
h = (h - h.min()) / (h.max() - h.min())
H = np.clip((h * LEVELS).astype(int), 0, LEVELS - 1)

SEA = 5
H = np.maximum(H, SEA)                      # flatten everything below sea level

# top colour by height band, sides are darker shades of it
def band(z):
    if z <= SEA:      return "#4A80B4", "#3C6A97", "#33597F"   # water
    if z <= SEA + 1:  return "#D9CBA0", "#BFB189", "#A79A76"   # sand
    if z <= LEVELS - 5: return "#6FA84A", "#5A8B3C", "#4A7331" # grass
    if z <= LEVELS - 3: return "#8A7355", "#725F46", "#5E4E3A" # dirt / rock
    return "#EDEDED", "#CFCFCF", "#B5B5B5"                     # snow

def iso(x, y, z):
    return ((x - y) * 0.866, -(x + y) * 0.5 + z * 0.8)

verts, cols = [], []
order = sorted(((x, y) for x in range(N) for y in range(N)), key=lambda p: p[0] + p[1])
for x, y in order:
    z = int(H[x, y])
    top, s1, s2 = band(z)
    verts.append([iso(x, y, z), iso(x + 1, y, z), iso(x + 1, y + 1, z), iso(x, y + 1, z)])
    cols.append(top)
    verts.append([iso(x + 1, y, z), iso(x + 1, y + 1, z), iso(x + 1, y + 1, 0), iso(x + 1, y, 0)])
    cols.append(s1)
    verts.append([iso(x, y + 1, z), iso(x + 1, y + 1, z), iso(x + 1, y + 1, 0), iso(x, y + 1, 0)])
    cols.append(s2)

fig, ax = plt.subplots(figsize=(11, 6.4))
ax.add_collection(PolyCollection(verts, facecolors=cols,
                                 edgecolors=cols, linewidths=0.25))
allv = np.array([p for q in verts for p in q])
ax.set_xlim(allv[:, 0].min(), allv[:, 0].max())
ax.set_ylim(allv[:, 1].min(), allv[:, 1].max())
ax.set_aspect("equal"); ax.axis("off")
ax.set_position([0, 0, 1, 1])
fig.patch.set_alpha(0)
plt.savefig(os.path.join(OUT, "voxel_world.png"), dpi=170, transparent=True,
            bbox_inches="tight", pad_inches=0)
plt.close()
print("voxel_world.png  (%d x %d columns, %d block levels, %d noise layers)"
      % (N, N, LEVELS, LAYERS))
