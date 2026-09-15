# Recolour the pre-baked violet artwork (Sierpinski / Koch / tree) onto the
# theme's clay hue. These PNGs have their colours baked in, so no preamble
# change can reach them. Hue is replaced; saturation and lightness are kept,
# so the pale interior tints stay pale.
#
# Originals are git-tracked: `git checkout -- images/` restores them.
import numpy as np
from PIL import Image
import colorsys, glob, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TARGET = "#A9745B"                      # ObjectColor in the themed preamble
LO, HI = 200.0 / 360.0, 330.0 / 360.0   # violet/blue band to replace

r, g, b = (int(TARGET[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
TH = colorsys.rgb_to_hsv(r, g, b)[0]

def recolor(path):
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).astype(np.float32) / 255.0
    rgb, alpha = a[..., :3], a[..., 3:]

    mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    ri, gi, bi = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    im_r = nz & (mx == ri); im_g = nz & (mx == gi); im_b = nz & (mx == bi)
    h[im_r] = ((gi - bi)[im_r] / d[im_r]) % 6
    h[im_g] = ((bi - ri)[im_g] / d[im_g]) + 2
    h[im_b] = ((ri - gi)[im_b] / d[im_b]) + 4
    h = h / 6.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    v = mx

    hit = nz & (h >= LO) & (h <= HI)
    if not hit.any():
        return False
    h = np.where(hit, TH, h)

    i = np.floor(h * 6.0); f = h * 6.0 - i
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    i = (i % 6).astype(np.int32)
    out = np.select(
        [(i == k)[..., None] for k in range(6)],
        [np.stack([v, t, p], -1), np.stack([q, v, p], -1), np.stack([p, v, t], -1),
         np.stack([p, q, v], -1), np.stack([t, p, v], -1), np.stack([v, p, q], -1)])
    out = np.where(hit[..., None], out, rgb)
    res = np.concatenate([out, alpha], -1)
    Image.fromarray((np.clip(res, 0, 1) * 255).astype(np.uint8)).save(path)
    return True

n = 0
for sub in ("sierpinski", "koch", "tree"):
    for f in sorted(glob.glob(os.path.join(ROOT, "images", sub, "*.png"))):
        if recolor(f):
            n += 1
            print("  recoloured", os.path.relpath(f, ROOT).replace("\\", "/"))
print("%d files recoloured to %s" % (n, TARGET))
