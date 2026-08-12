"""Procedural textures for the floor and the cube.

Runs under Isaac's CPython 3.11 (numpy + PIL only) and writes PNGs next to the
generated scene. They are build artifacts, not source: a seed in
``task/pick_cube.task.yaml`` makes them reproducible, so the bytes do not belong
in git.

**Why textures at all.** A policy that only sees images has to recover position
and orientation from those images, and a flat-shaded scene gives it nothing to
work with:

* An untextured floor has no landmarks. Every patch looks like every other
  patch, so there is no visual reference against which to place the cube or the
  gripper -- the network is left inferring depth from the arm's own silhouette.
* A uniformly red cube is **rotationally symmetric on camera**. ``spawn.yaw_deg``
  randomises the cube's yaw by ±45°, and with no surface detail that variation
  is literally invisible: two episodes that differ only in yaw produce identical
  pixels but different expert actions. That is label noise, and it is the kind
  that cannot be fixed by collecting more data.

So the floor gets large-scale irregular mottling (landmarks without a repeating
pattern that could alias with position), and the cube gets **a different
asymmetric pattern on each of its six faces** (orientation becomes readable).

⚠️ Both are deliberately low-saturation apart from the cube's red base. The cube
has to stay the most saturated red thing in frame -- that is what makes it
findable at the 8-12 px it occupies at the dataset's 320x240.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _value_noise(rng, size, octaves=5, persistence=0.5):
    """Multi-octave value noise in [0, 1].

    Built by upsampling small random grids rather than with a gradient-noise
    implementation: this only needs "irregular and non-repeating at the scale of
    the workspace", and bicubic upsampling of white noise gives exactly that in
    a few lines.
    """
    out = np.zeros((size, size), dtype=np.float64)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = 2 ** (o + 2)
        grid = (rng.random((n, n)) * 255).astype(np.uint8)
        up = np.asarray(Image.fromarray(grid).resize((size, size), Image.BICUBIC),
                        dtype=np.float64) / 255.0
        out += up * amp
        total += amp
        amp *= persistence
    out /= total
    # np.ptp(out), not out.ptp(): the method was removed in numpy 2.0, and this
    # repo already has both 1.26 (Isaac, ROS) and 2.x (lerobot) in play.
    return (out - out.min()) / max(float(np.ptp(out)), 1e-6)


def wood_texture(path, seed=0, size=1024, light=(0.87, 0.76, 0.58),
                 dark=(0.66, 0.53, 0.37), rings=11.0, distortion=1.3, knots=5,
                 sharpness=1.6):
    """Irregular wood grain for the table top.

    ⚠️ **Chosen for hue separation, not for looks.** The arm and gripper are grey
    metal, and a grey table put them at nearly the same lightness -- worst on the
    wrist camera, where the gripper fills most of the frame against a background
    of the same colour, so the policy has to segment grey from grey. Warm brown
    against cold grey separates in *hue*, which survives shading, shadows and
    exposure changes in a way a lightness difference does not.

    ⚠️ The separation from the cube is **saturation, not lightness.** Light oak
    (0.87, 0.76, 0.58) is about as bright as the cube's red (0.85, 0.15, 0.15) --
    in greyscale they nearly match. What keeps the cube findable at the 8-12 px
    it occupies in a 320x240 frame is that its saturation is ~0.82 against the
    wood's ~0.33. Darkening the wood would be the wrong fix if the cube ever gets
    lost; desaturating it is the right one.

    The grain runs along one axis -- real boards do -- but each ring is warped by
    noise and broken by knots, so the pattern never repeats and is not symmetric.
    """
    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, size)[:, None] * np.ones((1, size))

    # Rings: a sine along one axis, warped by noise so no line is straight.
    warp = _value_noise(rng, size, octaves=5)
    fine = _value_noise(rng, size, octaves=7, persistence=0.62)
    phase = yy * rings + warp * distortion

    # Boards. Real flooring is laid in strips, and this does more than look
    # right: each board carries its own grain phase and tint, so the pattern
    # **breaks** at every seam. That gives the strongest, least ambiguous
    # landmarks in the texture, and because the widths are drawn at random the
    # spacing is not a periodic signal.
    board_id = np.zeros((size, size), dtype=np.int32)
    edges, y = [0], 0
    while y < size:
        y += int(rng.uniform(0.11, 0.23) * size)
        edges.append(min(y, size))
    for i in range(len(edges) - 1):
        board_id[edges[i]:edges[i + 1], :] = i
    n_boards = len(edges) - 1
    phase = phase + (rng.random(n_boards) * 7.0)[board_id]

    grain = 0.5 + 0.5 * np.sin(phase * 2.0 * np.pi)
    # Sharpen: real grain has narrow dark lines, not a smooth sinusoid.
    grain = grain ** sharpness
    grain = np.clip(grain * 0.80 + fine * 0.20, 0.0, 1.0)
    # Per-board tint, then a dark line in the seam itself.
    grain *= (0.80 + 0.24 * rng.random(n_boards))[board_id]
    for e in edges[1:-1]:
        grain[max(0, e - 2):e + 2, :] *= 0.55

    # Knots -- the main source of large, obviously asymmetric structure.
    for _ in range(knots):
        cy, cx = rng.random(2) * size
        rad = rng.uniform(size * 0.02, size * 0.055)
        ry, rx = np.ogrid[:size, :size]
        d = np.sqrt(((ry - cy) / (rad * rng.uniform(1.4, 2.6))) ** 2
                    + ((rx - cx) / rad) ** 2)
        ring = np.clip(1.0 - d, 0.0, 1.0)
        grain = np.clip(grain - ring * 0.38 * (0.5 + 0.5 * np.sin(d * 26.0)), 0.0, 1.0)

    img = np.zeros((size, size, 3), dtype=np.float64)
    for c in range(3):
        img[:, :, c] = dark[c] + (light[c] - dark[c]) * grain
    Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8)).save(path)
    return path


def floor_texture(path, seed=0, size=1024, base=(0.46, 0.45, 0.44),
                  contrast=0.30, speckles=380, speckle_radius=(2, 10)):
    """Irregular mottled table top. One tile covers the whole quad -- see below.

    Superseded by ``wood_texture`` as the default (see there for why hue matters
    more than pattern), kept because it is a genuinely different visual domain
    and M5 may want to compare policies across both.

    ⚠️ The material must **not** tile this. A repeating surface gives the policy
    a periodic signal that correlates with position, which is worse than no
    landmarks at all: it looks informative and aliases every tile-width.

    Sizing matters more than it looks. The quad is 2 m across and this image is
    1024 px, so one texture pixel is ~2 mm; a camera at 0.5 m with a 90° FOV
    sees ~1 m across 320 px, i.e. ~3 mm per image pixel. Anything smaller than a
    few texture pixels is therefore **gone** by the time the policy sees it --
    which is why the speckles are drawn at 2-10 px and not 1-5.
    """
    rng = np.random.default_rng(seed)
    n = _value_noise(rng, size, octaves=6)
    # Two scales: broad patches, plus a finer grain so close-up views (the wrist
    # camera at 0.1 m) still have something to lock onto.
    fine = _value_noise(rng, size, octaves=3, persistence=0.65)
    shade = 1.0 + contrast * (n - 0.5) * 2.0 + 0.06 * (fine - 0.5) * 2.0

    img = np.zeros((size, size, 3), dtype=np.float64)
    for c in range(3):
        img[:, :, c] = base[c] * shade
    arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)

    # A few broad, faint stains -- large-scale structure that survives
    # downsampling and breaks the "uniform field of dots" look.
    #
    # ⚠️ Drawn as blurred ellipses, not as crop/paste rectangles. A rectangle is
    # axis-aligned by construction, so it puts perfectly horizontal and vertical
    # edges into the image -- a regular, human-made feature in a texture whose
    # entire purpose is to be irregular.
    stain = Image.new("L", (size, size), 0)
    sdraw = ImageDraw.Draw(stain)
    signs = []
    for _ in range(max(6, speckles // 40)):
        cx, cy = rng.integers(0, size, 2)
        rx, ry = rng.integers(size // 14, size // 5, 2)
        v = int(rng.integers(90, 230))
        sdraw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=v)
        signs.append(1 if rng.random() < 0.45 else -1)
    stain = stain.filter(ImageFilter.GaussianBlur(size // 45))
    s = np.asarray(stain, dtype=np.float64) / 255.0
    arr = np.asarray(im, dtype=np.float64)
    # One global sign would tint every stain the same way; alternating keeps
    # some lighter and some darker than the surface.
    arr += (s * 30.0)[:, :, None] * (1 if sum(signs) >= 0 else -1)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(im)
    # Asymmetric specks. These are the actual landmarks -- noise alone is smooth
    # enough that a downsampled 320x240 view washes it out.
    lo, hi = speckle_radius
    for _ in range(speckles):
        x, y = rng.integers(0, size, 2)
        r = int(rng.integers(lo, hi + 1))
        v = int(rng.integers(38, 105)) if rng.random() < 0.55 else int(rng.integers(150, 205))
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=(v, v, int(np.clip(v + rng.integers(-8, 9), 0, 255))))
    im.save(path)
    return path


# The cube's six faces, laid out 3 across and 2 down. Face order matches the
# order the mesh authors them in: +X -X +Y -Y +Z -Z.
CUBE_ATLAS_COLS, CUBE_ATLAS_ROWS = 3, 2


def cube_texture(path, seed=0, cell=256, base=(0.85, 0.15, 0.15)):
    """Six faces, each with its own asymmetric mark.

    Every face gets a different count, colour and placement of shapes, and each
    shape is placed off-centre, so the cube's orientation is recoverable from any
    view that sees a face -- including which way up it is.
    """
    rng = np.random.default_rng(seed + 1)
    W, H = cell * CUBE_ATLAS_COLS, cell * CUBE_ATLAS_ROWS
    im = Image.new("RGB", (W, H), tuple(int(c * 255) for c in base))
    draw = ImageDraw.Draw(im)

    for face in range(6):
        ox = (face % CUBE_ATLAS_COLS) * cell
        oy = (face // CUBE_ATLAS_COLS) * cell
        # Per-face tint so even a face seen without its marks is distinguishable.
        tint = tuple(int(np.clip(c * 255 * (0.78 + 0.09 * face), 0, 255)) for c in base)
        draw.rectangle([ox, oy, ox + cell - 1, oy + cell - 1], fill=tint)

        for _ in range(2 + face % 3):
            light = rng.random() < 0.5
            fill = (235, 228, 222) if light else (70, 22, 22)
            x0 = ox + int(rng.integers(int(cell * 0.08), int(cell * 0.55)))
            y0 = oy + int(rng.integers(int(cell * 0.08), int(cell * 0.55)))
            w = int(rng.integers(int(cell * 0.18), int(cell * 0.38)))
            h = int(rng.integers(int(cell * 0.18), int(cell * 0.38)))
            kind = int(rng.integers(0, 3))
            if kind == 0:
                draw.ellipse([x0, y0, x0 + w, y0 + h], fill=fill)
            elif kind == 1:
                draw.rectangle([x0, y0, x0 + w, y0 + h], fill=fill)
            else:
                draw.polygon([(x0, y0 + h), (x0 + w // 2, y0), (x0 + w, y0 + h)],
                             fill=fill)
        # A corner tick: gives an unambiguous "up" within the face, so in-plane
        # rotation is readable and not just which face is showing.
        t = int(cell * 0.11)
        draw.rectangle([ox + 5, oy + 5, ox + 5 + t, oy + 5 + t // 2], fill=(245, 245, 240))
    im.save(path)
    return path


def cube_face_uvs():
    """Face-varying UVs mapping each of the 6 quads to its atlas cell.

    Returned in the same face order the mesh is authored in. Each quad's four
    corners are given counter-clockwise, matching the vertex order in
    ``build_scene.box_mesh_points``.
    """
    uvs = []
    for face in range(6):
        c, r = face % CUBE_ATLAS_COLS, face // CUBE_ATLAS_COLS
        u0, u1 = c / CUBE_ATLAS_COLS, (c + 1) / CUBE_ATLAS_COLS
        # v is flipped: image row 0 is the top, UV v=0 is the bottom.
        v1, v0 = 1.0 - r / CUBE_ATLAS_ROWS, 1.0 - (r + 1) / CUBE_ATLAS_ROWS
        inset = 0.002          # keep bilinear filtering from bleeding across cells
        uvs += [(u0 + inset, v0 + inset), (u1 - inset, v0 + inset),
                (u1 - inset, v1 - inset), (u0 + inset, v1 - inset)]
    return uvs


# ── 三個具名物體 ────────────────────────────────────────────────────────
# 為什麼是這三個而不是三種顏色的方塊:純顏色的話,策略只要學「找最紅的像素」
# 就能過關,那離語言接地還很遠。這三個**都是白底或多彩、都是盒狀**,要分辨得
# 靠紋理樣式,而不是單一色相 —— 那才是要測的東西。
#
# ⚠️ 全部維持平面盒狀。物理跟現在那顆 25 mm 方塊完全相同(專家 analytic 20/20),
# 這一輪只改任務語意,不動已經調好的接觸幾何。
#
# 低解析度下的判別依據(方塊在 448x336 下約 15 px 邊長):
#   dice    高明度、低飽和、稀疏黑點
#   rubik   多色相、高飽和、高空間頻率
#   eraser  單一粉色、低空間頻率、比例扁長
# 三者在「明度 / 飽和 / 空間頻率」三個軸上互相分開,不依賴單一線索。

def dice_texture(path, seed=0, cell=256):
    """White die, black pips, faces 1-6 in the standard opposite-sums-to-7 layout."""
    rng = np.random.default_rng(seed + 11)
    W, H = cell * CUBE_ATLAS_COLS, cell * CUBE_ATLAS_ROWS
    im = Image.new("RGB", (W, H), (242, 240, 235))
    draw = ImageDraw.Draw(im)
    # (col, row) pip positions on a 3x3 grid, per face value.
    PIPS = {
        1: [(1, 1)],
        2: [(0, 0), (2, 2)],
        3: [(0, 0), (1, 1), (2, 2)],
        4: [(0, 0), (2, 0), (0, 2), (2, 2)],
        5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
        6: [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)],
    }
    r = cell * 0.115                      # pip radius -- large, so it survives 15 px
    for face in range(6):
        ox, oy = (face % CUBE_ATLAS_COLS) * cell, (face // CUBE_ATLAS_COLS) * cell
        # faint ivory shading so faces are not bit-identical under flat light
        draw.rectangle([ox, oy, ox + cell - 1, oy + cell - 1],
                       fill=(242 - face, 240 - face, 233 - face))
        for c, rr in PIPS[face + 1]:
            cx = ox + cell * (0.26 + 0.24 * c)
            cy = oy + cell * (0.26 + 0.24 * rr)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(24, 22, 20))
        # rounded-corner hint: dark corner ticks read as a die's bevel at distance
        t = int(cell * 0.06)
        for dx, dy in ((0, 0), (cell - t, 0), (0, cell - t), (cell - t, cell - t)):
            draw.rectangle([ox + dx, oy + dy, ox + dx + t, oy + dy + t],
                           fill=(214, 210, 202))
    im.save(path)
    return path


def rubik_texture(path, seed=0, cell=256):
    """3x3 sticker grid per face -- reads as "multicoloured" long before the
    individual stickers resolve, which is the point at 15 px."""
    rng = np.random.default_rng(seed + 22)
    FACE = [(200, 30, 30), (30, 120, 220), (240, 240, 235),
            (245, 200, 20), (20, 150, 70), (240, 120, 20)]
    W, H = cell * CUBE_ATLAS_COLS, cell * CUBE_ATLAS_ROWS
    im = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(im)
    g = cell * 0.055                      # black gap between stickers
    s = (cell - 4 * g) / 3
    for face in range(6):
        ox, oy = (face % CUBE_ATLAS_COLS) * cell, (face // CUBE_ATLAS_COLS) * cell
        draw.rectangle([ox, oy, ox + cell - 1, oy + cell - 1], fill=(18, 18, 20))
        for i in range(3):
            for j in range(3):
                # centre sticker keeps the face colour; the rest are scrambled,
                # so no face is a flat colour patch that could pass for a plain cube
                col = FACE[face] if (i == 1 and j == 1) else FACE[int(rng.integers(0, 6))]
                x0 = ox + g + i * (s + g); y0 = oy + g + j * (s + g)
                draw.rounded_rectangle([x0, y0, x0 + s, y0 + s],
                                       radius=s * 0.18, fill=col)
    im.save(path)
    return path


def eraser_texture(path, seed=0, cell=256):
    """Blue block with a white paper sleeve across the middle.

    The sleeve is the low-frequency cue that survives downsampling.

    ⚠️ **Blue, not the more obvious pink.** Measured at 15 px, a pink eraser sits
    at S=0.25 against a wood floor at S=0.28 and only ~40 deg of hue away -- it
    would have to be segmented on brightness alone, which drifts with shadow.
    Blue puts ~170 deg between them and roughly doubles the saturation gap, while
    staying well clear of the Rubik's cube on edge density (14 vs 62).
    """
    rng = np.random.default_rng(seed + 33)
    PINK = (62, 118, 200)
    W, H = cell * CUBE_ATLAS_COLS, cell * CUBE_ATLAS_ROWS
    im = Image.new("RGB", (W, H), PINK)
    draw = ImageDraw.Draw(im)
    for face in range(6):
        ox, oy = (face % CUBE_ATLAS_COLS) * cell, (face // CUBE_ATLAS_COLS) * cell
        draw.rectangle([ox, oy, ox + cell - 1, oy + cell - 1],
                       fill=(PINK[0] - face * 3, PINK[1] - face * 2, PINK[2] - face * 2))
        # paper sleeve -- a broad light band, placed off-centre so the face's
        # orientation is still recoverable
        band = cell * (0.30 + 0.05 * (face % 3))
        y0 = oy + cell * (0.22 + 0.09 * (face % 4))
        draw.rectangle([ox, y0, ox + cell - 1, y0 + band], fill=(246, 243, 236))
        draw.rectangle([ox, y0, ox + cell - 1, y0 + cell * 0.02], fill=(120, 118, 112))
        draw.rectangle([ox, y0 + band, ox + cell - 1, y0 + band + cell * 0.02],
                       fill=(120, 118, 112))
        # a couple of scuffs so it is not a perfectly clean synthetic block
        for _ in range(3):
            x = ox + int(rng.integers(0, cell)); y = oy + int(rng.integers(0, cell))
            d = int(cell * 0.05)
            draw.ellipse([x, y, x + d, y + d], fill=(48, 96, 170))
    im.save(path)
    return path
