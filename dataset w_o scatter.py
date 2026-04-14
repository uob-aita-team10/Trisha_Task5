"""
generate_dataset_v2.py  —  Alternative Option 1a dataset generator

Key differences from generate_dataset.py:
  1. Proper filled geometric shapes via matplotlib.patches (Circle, FancyBboxPatch,
     Polygon, RegularPolygon) — objects are visually distinct, not scatter markers.
  2. Five varied sentence templates per binary scene — sentences no longer all share
     the same grammar structure (addresses the brief's "ideally sentences won't all
     have the same structure" requirement).
  3. Soft random background colours (very light pastels) — the model cannot exploit
     a constant white background as a shortcut feature.
  4. Duplicate-free binary scenes — a (size1,colour1,shape1,relation,size2,colour2,
     shape2) combination is only generated once.
  5. Vocabulary kept identical to dataset.py so the rest of the pipeline
     (dataset.py / model.py / train.py / evaluate.py) works without any changes.

Usage:
    python generate_dataset_v2.py

Output is written to  data_v2/  in the working directory:
    data_v2/train.json
    data_v2/val.json
    data_v2/test.json
    data_v2/images/img_XXXXXX.png
    data_v2/dataset_stats.json
"""

import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Vocabulary (must match dataset.py exactly) ─────────────────────────────────

SIZES = ["small", "medium", "big"]
COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
SHAPES = ["sphere", "cube", "triangle", "diamond", "star", "pentagon", "hexagon", "cross"]
RELATIONS = [
    "above", "below", "to the left of", "to the right of",
    "upper left of", "upper right of", "lower left of", "lower right of",
]

# ── Colour hex values ──────────────────────────────────────────────────────────

COLOUR_HEX = {
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "green":  "#2ecc71",
    "yellow": "#f1c40f",
    "purple": "#9b59b6",
    "orange": "#e67e22",
    "pink":   "#e91e8b",
    "cyan":   "#00bcd4",
}

# ── Size → radius in axis units (axes run 0–1) ─────────────────────────────────

SIZE_RADIUS = {
    "small":  0.055,
    "medium": 0.085,
    "big":    0.120,
}

# ── Spatial offsets (object-1 relative to object-2) ───────────────────────────

DIRECTIONS = {
    "above":           ( 0.00,  0.28),
    "below":           ( 0.00, -0.28),
    "to the left of":  (-0.28,  0.00),
    "to the right of": ( 0.28,  0.00),
    "upper left of":   (-0.22,  0.22),
    "upper right of":  ( 0.22,  0.22),
    "lower left of":   (-0.22, -0.22),
    "lower right of":  ( 0.22, -0.22),
}

# ── Image dimensions ───────────────────────────────────────────────────────────

IMG_SIZE   = 128          # pixels
DPI        = 64
FIG_INCHES = IMG_SIZE / DPI   # 2.0 inches

# ── Background palettes (very light pastels so shapes stay vivid) ──────────────

BG_COLOURS = [
    "#f5f5f5",   # near-white
    "#fdf6e3",   # warm cream
    "#eaf4fb",   # pale blue
    "#eafaf1",   # pale green
    "#fdf2f8",   # pale pink
    "#fef9e7",   # pale yellow
    "#f4ecf7",   # pale lavender
    "#e8f8f5",   # pale teal
]

# ── Sentence templates ─────────────────────────────────────────────────────────
# Each template is a callable: f(size1,colour1,shape1,relation,size2,colour2,shape2) → str

TEMPLATES = [
    # Template 0  — original style
    lambda s1, c1, sh1, rel, s2, c2, sh2:
        f"a {s1} {c1} {sh1} is {rel} a {s2} {c2} {sh2}",

    # Template 1  — "the … sits …"
    lambda s1, c1, sh1, rel, s2, c2, sh2:
        f"the {s1} {c1} {sh1} sits {rel} the {s2} {c2} {sh2}",

    # Template 2  — "there is a … [relation] a …"
    lambda s1, c1, sh1, rel, s2, c2, sh2:
        f"there is a {s1} {c1} {sh1} {rel} a {s2} {c2} {sh2}",

    # Template 3  — "a [colour] [shape] ([size]) is positioned …"
    lambda s1, c1, sh1, rel, s2, c2, sh2:
        f"a {c1} {sh1} ({s1}) is positioned {rel} a {c2} {sh2} ({s2})",

    # Template 4  — "you can see a … placed …"
    lambda s1, c1, sh1, rel, s2, c2, sh2:
        f"you can see a {s1} {c1} {sh1} placed {rel} a {s2} {c2} {sh2}",
]


# ── Shape drawing ──────────────────────────────────────────────────────────────

def _make_star_polygon(cx, cy, r, n=5):
    """Return x,y arrays for a filled n-pointed star centred at (cx,cy)."""
    outer, inner = r, r * 0.4
    angles = [math.pi / 2 + 2 * math.pi * k / (2 * n) for k in range(2 * n)]
    radii  = [outer if k % 2 == 0 else inner for k in range(2 * n)]
    xs = [cx + radii[k] * math.cos(angles[k]) for k in range(2 * n)]
    ys = [cy + radii[k] * math.sin(angles[k]) for k in range(2 * n)]
    return xs, ys


def draw_shape(ax, shape_name, colour_name, size_name, cx, cy):
    """Draw a proper filled geometric shape on *ax* centred at (cx, cy)."""
    r    = SIZE_RADIUS[size_name]
    hex_ = COLOUR_HEX[colour_name]
    ec   = "black"
    lw   = 0.8

    if shape_name == "sphere":
        patch = mpatches.Circle((cx, cy), r, color=hex_, ec=ec, lw=lw, zorder=5)
        ax.add_patch(patch)

    elif shape_name == "cube":
        side = r * 1.6
        patch = mpatches.FancyBboxPatch(
            (cx - side / 2, cy - side / 2), side, side,
            boxstyle="square,pad=0", color=hex_, ec=ec, lw=lw, zorder=5,
        )
        ax.add_patch(patch)

    elif shape_name == "triangle":
        h   = r * 1.8
        pts = np.array([
            [cx,         cy + h * 2 / 3],
            [cx - h / math.sqrt(3), cy - h / 3],
            [cx + h / math.sqrt(3), cy - h / 3],
        ])
        patch = mpatches.Polygon(pts, closed=True, color=hex_, ec=ec, lw=lw, zorder=5)
        ax.add_patch(patch)

    elif shape_name == "diamond":
        pts = np.array([
            [cx,       cy + r * 1.5],
            [cx + r,   cy],
            [cx,       cy - r * 1.5],
            [cx - r,   cy],
        ])
        patch = mpatches.Polygon(pts, closed=True, color=hex_, ec=ec, lw=lw, zorder=5)
        ax.add_patch(patch)

    elif shape_name == "star":
        xs, ys = _make_star_polygon(cx, cy, r * 1.5, n=5)
        pts = np.column_stack([xs, ys])
        patch = mpatches.Polygon(pts, closed=True, color=hex_, ec=ec, lw=lw, zorder=5)
        ax.add_patch(patch)

    elif shape_name == "pentagon":
        patch = mpatches.RegularPolygon(
            (cx, cy), numVertices=5, radius=r * 1.3,
            orientation=math.pi / 2,
            color=hex_, ec=ec, lw=lw, zorder=5,
        )
        ax.add_patch(patch)

    elif shape_name == "hexagon":
        patch = mpatches.RegularPolygon(
            (cx, cy), numVertices=6, radius=r * 1.2,
            orientation=0,
            color=hex_, ec=ec, lw=lw, zorder=5,
        )
        ax.add_patch(patch)

    elif shape_name == "cross":
        t = r * 0.45          # arm half-thickness
        arm = r * 1.4         # arm half-length
        # Horizontal bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - arm, cy - t), arm * 2, t * 2,
            boxstyle="square,pad=0", color=hex_, ec=ec, lw=lw, zorder=5,
        ))
        # Vertical bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - t, cy - arm), t * 2, arm * 2,
            boxstyle="square,pad=0", color=hex_, ec=ec, lw=lw, zorder=5,
        ))


# ── Image rendering ────────────────────────────────────────────────────────────

def render_image(objects, save_path, bg_colour="#f5f5f5"):
    fig, ax = plt.subplots(1, 1, figsize=(FIG_INCHES, FIG_INCHES), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(bg_colour)
    fig.patch.set_facecolor(bg_colour)

    for obj in objects:
        draw_shape(ax, obj["object"], obj["color"], obj["size"], obj["x"], obj["y"])

    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def clamp(v, lo=0.15, hi=0.85):
    """Keep coordinates away from image borders so shapes are not clipped."""
    return max(lo, min(hi, v))


def add_noise(v, scale=0.03):
    return v + random.uniform(-scale, scale)


# ── Sample generators ──────────────────────────────────────────────────────────

def generate_binary_sample(idx, used_combos):
    """
    Generate one binary scene.
    Retries until a (size1,colour1,shape1,relation,size2,colour2,shape2) combo
    that has not been used before is found.
    Returns the sample dict and the updated used_combos set.
    """
    for _ in range(2000):   # guard against infinite loops if pool is exhausted
        s1  = random.choice(SIZES)
        c1  = random.choice(COLOURS)
        sh1 = random.choice(SHAPES)
        rel = random.choice(RELATIONS)
        s2  = random.choice(SIZES)
        c2  = random.choice(COLOURS)
        sh2 = random.choice(SHAPES)

        key = (s1, c1, sh1, rel, s2, c2, sh2)
        if key in used_combos:
            continue
        used_combos.add(key)

        # Pick sentence template at random
        template = random.choice(TEMPLATES)
        sentence = template(s1, c1, sh1, rel, s2, c2, sh2)

        # Place object 2 near centre; object 1 offset by relation vector
        dx, dy = DIRECTIONS[rel]
        cx2 = clamp(add_noise(0.5, 0.08))
        cy2 = clamp(add_noise(0.5, 0.08))
        cx1 = clamp(add_noise(cx2 + dx, 0.03))
        cy1 = clamp(add_noise(cy2 + dy, 0.03))

        objects = [
            {"object": sh1, "color": c1, "size": s1, "x": cx1, "y": cy1},
            {"object": sh2, "color": c2, "size": s2, "x": cx2, "y": cy2},
        ]

        bg = random.choice(BG_COLOURS)

        return {
            "id":       idx,
            "type":     "binary",
            "sentence": sentence,
            "direction": rel,
            "objects":  objects,
            "bg":       bg,
        }, used_combos

    raise RuntimeError("Combinatorial pool exhausted — reduce N_BINARY.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(0)
    np.random.seed(0)

    output_dir = Path(__file__).parent / "data_v2"
    image_dir  = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Maximum unique combos = 3*8*8*8*3*8*8 = 294,912 — plenty of headroom
    N_BINARY = 15_000
    TOTAL    = N_BINARY

    dataset    = []
    used_combos = set()
    idx        = 0

    print(f"\nGenerating {N_BINARY} unique binary scenes …")
    for i in range(N_BINARY):
        sample, used_combos = generate_binary_sample(idx, used_combos)
        img_filename = f"img_{idx:06d}.png"
        sample["image"] = f"images/{img_filename}"
        render_image(sample["objects"], image_dir / img_filename, bg_colour=sample["bg"])
        dataset.append(sample)
        idx += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N_BINARY} done")

    # ── Split 80 / 10 / 10 ────────────────────────────────────────────────────
    random.shuffle(dataset)

    n_train = int(TOTAL * 0.8)
    n_val   = int(TOTAL * 0.1)

    train_set = dataset[:n_train]
    val_set   = dataset[n_train : n_train + n_val]
    test_set  = dataset[n_train + n_val :]

    for split_data in (train_set, val_set, test_set):
        for i, s in enumerate(split_data):
            s["id"] = i

    for split_name, split_data in [("train", train_set),
                                    ("val",   val_set),
                                    ("test",  test_set)]:
        json_path = output_dir / f"{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"  {split_name}.json : {len(split_data)} samples")

    # ── Template distribution report ──────────────────────────────────────────
    template_counts = {}
    for s in dataset:
        # Identify template by first distinctive word
        sent = s["sentence"]
        if sent.startswith("a "):
            key = "T0: a … is …"
        elif sent.startswith("the "):
            key = "T1: the … sits …"
        elif sent.startswith("there"):
            key = "T2: there is a …"
        elif sent[2:3] == " " or "(" in sent:
            key = "T3: a colour shape (size) …"
        else:
            key = "T4: you can see …"
        template_counts[key] = template_counts.get(key, 0) + 1

    print("\nTemplate distribution:")
    for k, v in sorted(template_counts.items()):
        print(f"  {k}: {v}")

    # ── Stats JSON ────────────────────────────────────────────────────────────
    stats = {
        "total_samples":  TOTAL,
        "train_count":    len(train_set),
        "val_count":      len(val_set),
        "test_count":     len(test_set),
        "binary_count":   N_BINARY,
        "unique_combos":  len(used_combos),
        "shapes":         SHAPES,
        "colours":        COLOURS,
        "sizes":          SIZES,
        "relations":      RELATIONS,
        "num_templates":  len(TEMPLATES),
        "image_size_px":  IMG_SIZE,
        "rendering":      "matplotlib.patches (proper filled polygons)",
    }
    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to: {output_dir}")
    print(f"  train.json : {len(train_set)} samples (80%)")
    print(f"  val.json   : {len(val_set)} samples (10%)")
    print(f"  test.json  : {len(test_set)} samples (10%)")
    print(f"  images/    : {TOTAL} PNG files")
    print(f"  Unique scene combos used: {len(used_combos)}")


if __name__ == "__main__":
    main()
