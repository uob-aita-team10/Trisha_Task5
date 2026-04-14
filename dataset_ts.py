"""
Dataset.py

Generates the scene description dataset: renders PNG images of simple geometric
scenes using matplotlib scatter markers and writes train / val / test JSON splits.

Three scene types are produced:
    single  — one object placed at a named screen position
    binary  — two objects with a spatial relation (above, left of, etc.)
    triple  — two objects flanking a third (left-and-right / top-and-bottom)

Split ratio: 80% train / 10% val / 10% test  (2000 samples total).

Changes in this version:
    - A list of soft pastel background colours (BG_COLORS) has been added.
    - render_image() now accepts an optional bg_color parameter so each image
      can have a different background instead of always being white.

"""

import json
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend so no window opens on screen
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. Shape, colour, size, and layout definitions
# ============================================================

# Maps each shape name to its matplotlib scatter marker code.
# These marker codes control how the shape looks when drawn with ax.scatter().

OBJECTS = {
    "sphere":   {"marker": "o"},
    "cube":     {"marker": "s"},
    "triangle": {"marker": "^"},
    "diamond":  {"marker": "D"},
    "star":     {"marker": "*"},
    "pentagon": {"marker": "p"},
    "hexagon":  {"marker": "h"},
    "cross":    {"marker": "P"},
}

# Maps each colour name to its hex colour code used when filling the shape.
COLORS = {
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "green":  "#2ecc71",
    "yellow": "#f1c40f",
    "purple": "#9b59b6",
    "orange": "#e67e22",
    "pink":   "#e91e8b",
    "cyan":   "#00bcd4",
}

# Maps each size name to a marker_size number used by ax.scatter().
SIZES = {
    "big":    {"marker_size": 1200},
    "medium": {"marker_size": 600},
    "small":  {"marker_size": 250},
}

# Maps each spatial relation to (dx, dy) offsets in canvas units (canvas is 0–1).
# dx = horizontal shift, dy = vertical shift.
# Object 1 is placed at (object2_x + dx, object2_y + dy).
#   Positive dy = upward,  negative dy = downward
#   Positive dx = rightward, negative dx = leftward
DIRECTIONS = {
    "above":           {"dx": 0.0,   "dy": 0.25},
    "below":           {"dx": 0.0,   "dy": -0.25},
    "to the left of":  {"dx": -0.25, "dy": 0.0},
    "to the right of": {"dx": 0.25,  "dy": 0.0},
    "upper left of":   {"dx": -0.20, "dy": 0.20},
    "upper right of":  {"dx": 0.20,  "dy": 0.20},
    "lower left of":   {"dx": -0.20, "dy": -0.20},
    "lower right of":  {"dx": 0.20,  "dy": -0.20},
}

# Maps each named screen position to its (x, y) canvas coordinate.
# Used only by single-object scenes to name where the object appears.
# Canvas runs from 0 to 1 in both directions; (0.5, 0.5) is the centre.
SCREEN_POSITIONS = {
    "center":       {"x": 0.50, "y": 0.50},
    "top":          {"x": 0.50, "y": 0.75},
    "bottom":       {"x": 0.50, "y": 0.25},
    "left":         {"x": 0.25, "y": 0.50},
    "right":        {"x": 0.75, "y": 0.50},
    "top left":     {"x": 0.25, "y": 0.75},
    "top right":    {"x": 0.75, "y": 0.75},
    "bottom left":  {"x": 0.25, "y": 0.25},
    "bottom right": {"x": 0.75, "y": 0.25},
}

# Maps each triple-scene layout name to the offsets of the two flanking objects
# relative to the centre object.
#   "left and right" — objects 1 and 2 are placed to the left and right of object 3
#   "top and bottom" — objects 1 and 2 are placed above and below object 3
BOTH_SIDES_LAYOUTS = {
    "left and right": {"left_dx": -0.28, "left_dy": 0.0,  "right_dx": 0.28, "right_dy": 0.0},
    "top and bottom": {"left_dx": 0.0,   "left_dy": 0.25, "right_dx": 0.0,  "right_dy": -0.25},
}

# ============================================================
# Background colour palette (new)
# ============================================================

# A list of very light pastel background colours.
# One is picked randomly for each image so the model cannot learn that the
# background is always white — a shortcut that would not generalise.
# All colours are deliberately pale so the coloured shapes remain vivid.
BG_COLORS = [
    "#f5f5f5",   # near-white
    "#fdf6e3",   # warm cream
    "#eaf4fb",   # pale blue
    "#eafaf1",   # pale green
    "#fdf2f8",   # pale pink
    "#fef9e7",   # pale yellow
    "#f4ecf7",   # pale lavender
    "#e8f8f5",   # pale teal
]

# ============================================================
# 2. Image drawing
# ============================================================

# Output image dimensions.
# IMG_SIZE / DPI = FIG_INCHES → matplotlib figure size in inches.
# Combined with DPI this produces exactly IMG_SIZE × IMG_SIZE pixels.
IMG_SIZE   = 128   # final image width and height in pixels
DPI        = 64    # dots per inch used when saving the figure
FIG_INCHES = IMG_SIZE / DPI   # figure size in inches (128 / 64 = 2.0)


def draw_shape(ax, obj_name, color_name, size_name, x, y):
    """
    Draw one scatter-marker shape onto the matplotlib axes *ax*.

    Looks up the marker code, hex colour, and marker size from the
    dictionaries above, then calls ax.scatter() to plot the shape at (x, y).

    Args:
        ax         : matplotlib Axes object to draw on
        obj_name   : shape name key from OBJECTS (e.g. "sphere")
        color_name : colour name key from COLORS  (e.g. "red")
        size_name  : size name key from SIZES     (e.g. "big")
        x, y       : canvas coordinates in [0, 1]
    """
    marker    = OBJECTS[obj_name]["marker"]    # matplotlib marker code e.g. "o"
    hex_color = COLORS[color_name]             # fill colour e.g. "#e74c3c"
    ms        = SIZES[size_name]["marker_size"] # scatter size in area units

    # Plot the shape as a single scatter point.
    # edgecolors="black" adds a thin black outline so shapes are clearly visible.
    # zorder=5 ensures shapes are drawn on top of the background.
    ax.scatter(x, y, marker=marker, s=ms, c=hex_color,
               edgecolors="black", linewidths=0.6, zorder=5)


def render_image(obj_list, save_path, bg_color="white"):
    """
    Create a 128×128 px canvas, draw all objects in obj_list, and save as PNG.

    Args:
        obj_list  : list of object dicts, each containing keys:
                    "object", "color", "size", "x", "y"
        save_path : full file path where the PNG will be saved
        bg_color  : background colour hex string or name (default "white").
                    Pass a value from BG_COLORS to use a pastel background.
    """
    # Create a square figure and axes at the target size
    fig, ax = plt.subplots(1, 1, figsize=(FIG_INCHES, FIG_INCHES), dpi=DPI)

    # Set coordinate system to run from 0 to 1 in both directions
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Keep the canvas perfectly square so shapes are not stretched
    ax.set_aspect("equal")

    # Hide axis lines and tick marks — the image should be clean
    ax.axis("off")

    # Apply the chosen background colour to both the plot area and outer figure
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    # Draw each object in the scene using draw_shape()
    for obj in obj_list:
        draw_shape(ax, obj["object"], obj["color"], obj["size"], obj["x"], obj["y"])

    # Save the finished figure as a PNG file and close it to free memory.
    # bbox_inches="tight" removes excess whitespace around the figure.
    # pad_inches=0.05 keeps a very small border so shapes at the edge are not cut.
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ============================================================
# 3. Sample generation
# ============================================================

def clamp(v, lo=0.08, hi=0.92):
    """
    Clamp value v between lo and hi.

    Keeps shape coordinates at least 0.08 units away from the canvas edges
    so shapes are never partially clipped at the border.
    """
    return max(lo, min(hi, v))


def add_noise(v, scale=0.04):
    """
    Add a small random offset (up to ±scale) to value v.

    Prevents every scene with the same relation from placing objects at
    exactly the same pixel positions, making each image slightly unique.
    """
    return v + random.uniform(-scale, scale)


def generate_binary_sample(idx):
    """
    Generate one binary scene — two objects with a spatial relation.

    Randomly picks a shape, colour, and size for each of the two objects and
    a random spatial relation between them.  Object 2 is placed near the
    canvas centre; object 1 is offset from object 2 using the relation vector.

    Sentence format:
        "a {size1} {color1} {obj1} is {relation} a {size2} {color2} {obj2}"

    Returns a sample dict (does not include the "image" or "bg" fields yet —
    those are added by the caller in main()).
    """
    # Randomly choose attributes for both objects and the spatial relation
    s1_obj    = random.choice(list(OBJECTS.keys()))
    s1_color  = random.choice(list(COLORS.keys()))
    s1_size   = random.choice(list(SIZES.keys()))
    s2_obj    = random.choice(list(OBJECTS.keys()))
    s2_color  = random.choice(list(COLORS.keys()))
    s2_size   = random.choice(list(SIZES.keys()))
    direction = random.choice(list(DIRECTIONS.keys()))

    # Build the natural language description of the scene
    sentence = (f"a {s1_size} {s1_color} {s1_obj} is {direction} "
                f"a {s2_size} {s2_color} {s2_obj}")

    # Place object 2 near the centre (0.5, 0.5) with small random noise
    cx, cy = 0.5, 0.5
    d      = DIRECTIONS[direction]
    s2_x   = clamp(add_noise(cx))
    s2_y   = clamp(add_noise(cy))

    # Place object 1 relative to object 2 using the direction offset, also with noise
    s1_x = clamp(add_noise(s2_x + d["dx"]))
    s1_y = clamp(add_noise(s2_y + d["dy"]))

    objects = [
        {"object": s1_obj, "color": s1_color, "size": s1_size, "x": s1_x, "y": s1_y},
        {"object": s2_obj, "color": s2_color, "size": s2_size, "x": s2_x, "y": s2_y},
    ]
    return {"id": idx, "type": "binary", "sentence": sentence,
            "direction": direction, "objects": objects}


def generate_single_sample(idx):
    """
    Generate one single-object scene — one shape at a named screen position.

    Randomly picks a shape, colour, size, and one of 9 named positions.
    Adds small noise to the exact coordinates so the object is never at
    exactly the same pixel every time the same position is chosen.

    Sentence format:
        "a {size} {color} {obj} appears at the {position} of the image"

    Returns a sample dict (does not include the "image" or "bg" fields yet).
    """
    # Randomly choose the object attributes and screen position
    obj_name   = random.choice(list(OBJECTS.keys()))
    color_name = random.choice(list(COLORS.keys()))
    size_name  = random.choice(list(SIZES.keys()))
    position   = random.choice(list(SCREEN_POSITIONS.keys()))

    # Build the natural language description
    sentence = (f"a {size_name} {color_name} {obj_name} appears "
                f"at the {position} of the image")

    # Look up the named position's coordinates and add small noise
    pos = SCREEN_POSITIONS[position]
    x   = clamp(add_noise(pos["x"]))
    y   = clamp(add_noise(pos["y"]))

    objects = [{"object": obj_name, "color": color_name, "size": size_name, "x": x, "y": y}]
    return {"id": idx, "type": "single", "sentence": sentence,
            "position": position, "objects": objects}


def generate_triple_sample(idx):
    """
    Generate one triple-object scene — two flanking objects and one centre object.

    Randomly picks 3 independent sets of shape/colour/size attributes and one
    of 2 layout arrangements.  The centre object (index 2) is placed near the
    canvas middle; the two flanking objects (indices 0 and 1) are offset from
    the centre using the layout offsets.

    Sentence format:
        "a {size0} {color0} {obj0} and a {size1} {color1} {obj1}
         are on the {layout} of a {size2} {color2} {obj2}"

    Returns a sample dict (does not include the "image" or "bg" fields yet).
    """
    # Randomly choose 3 shapes, 3 colours, 3 sizes, and a layout arrangement
    names      = [random.choice(list(OBJECTS.keys())) for _ in range(3)]
    colors     = [random.choice(list(COLORS.keys()))  for _ in range(3)]
    sizes      = [random.choice(list(SIZES.keys()))   for _ in range(3)]
    layout_key = random.choice(list(BOTH_SIDES_LAYOUTS.keys()))
    layout     = BOTH_SIDES_LAYOUTS[layout_key]

    # Build the natural language description
    sentence = (f"a {sizes[0]} {colors[0]} {names[0]} and "
                f"a {sizes[1]} {colors[1]} {names[1]} are on the "
                f"{layout_key} of a {sizes[2]} {colors[2]} {names[2]}")

    # Place the centre object (index 2) near (0.5, 0.5) with small noise
    cx, cy = 0.5, 0.5
    s3_x   = clamp(add_noise(cx, 0.03))
    s3_y   = clamp(add_noise(cy, 0.03))

    # Place the two flanking objects offset from the centre using layout offsets
    s1_x = clamp(add_noise(s3_x + layout["left_dx"],  0.03))
    s1_y = clamp(add_noise(s3_y + layout["left_dy"],  0.03))
    s2_x = clamp(add_noise(s3_x + layout["right_dx"], 0.03))
    s2_y = clamp(add_noise(s3_y + layout["right_dy"], 0.03))

    objects = [
        {"object": names[0], "color": colors[0], "size": sizes[0], "x": s1_x, "y": s1_y},
        {"object": names[1], "color": colors[1], "size": sizes[1], "x": s2_x, "y": s2_y},
        {"object": names[2], "color": colors[2], "size": sizes[2], "x": s3_x, "y": s3_y},
    ]
    return {"id": idx, "type": "triple", "sentence": sentence,
            "layout": layout_key, "objects": objects}


def main():
    """
    Main entry point — generates all samples, saves images and JSON splits.

    Steps:
        1. Generate N_SINGLE single-object scenes
        2. Generate N_BINARY binary scenes
        3. Generate N_TRIPLE triple scenes
        4. Split dataset into train (80%) / val (10%) / test (10%)
        5. Save each split as a JSON file
        6. Print sample examples and save a dataset_stats.json summary
    """
    # Fix random seeds so the dataset is identical every time the script is run
    random.seed(42)
    np.random.seed(42)

    # Create the output directory and the images subfolder inside it
    output_dir = Path(__file__).parent / "dataset"
    image_dir  = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)  # no error if folders already exist

    # --- Total 2000 samples, roughly balanced across three types ---
    N_SINGLE = 500
    N_BINARY = 1000
    N_TRIPLE = 500
    TOTAL    = N_SINGLE + N_BINARY + N_TRIPLE  # 650

    dataset = []   # will collect all generated sample dicts
    idx     = 0    # global counter used to name image files sequentially

    # Three scene types are generated one after another
    # --- Single-object scenes ---
    print(f"\n[1/3] Generating single-object samples ({N_SINGLE})...")
    for i in range(N_SINGLE):
        sample = generate_single_sample(idx)

        # Give the image a zero-padded filename e.g. img_000000.png
        img_filename    = f"img_{idx:06d}.png"
        sample["image"] = f"images/{img_filename}"

        # Pick a random pastel background for this image (new)
        bg = random.choice(BG_COLORS)
        sample["bg"] = bg   # store the chosen colour in the sample dict (new)

        # Draw and save the image to disk using the chosen background colour
        render_image(sample["objects"], image_dir / img_filename, bg_color=bg)

        dataset.append(sample)
        idx += 1

        # Print progress every 250 samples
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{N_SINGLE} done")

    # --- Binary scenes ---
    print(f"\n[2/3] Generating binary samples ({N_BINARY})...")
    for i in range(N_BINARY):
        sample = generate_binary_sample(idx)

        img_filename    = f"img_{idx:06d}.png"
        sample["image"] = f"images/{img_filename}"

        # Pick a random pastel background for this image (new)
        bg = random.choice(BG_COLORS)
        sample["bg"] = bg   # store the chosen colour in the sample dict (new)

        render_image(sample["objects"], image_dir / img_filename, bg_color=bg)

        dataset.append(sample)
        idx += 1

        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{N_BINARY} done")

    # --- Triple scenes ---
    print(f"\n[3/3] Generating triple samples ({N_TRIPLE})...")
    for i in range(N_TRIPLE):
        sample = generate_triple_sample(idx)

        img_filename    = f"img_{idx:06d}.png"
        sample["image"] = f"images/{img_filename}"

        # Pick a random pastel background for this image (new)
        bg = random.choice(BG_COLORS)
        sample["bg"] = bg   # store the chosen colour in the sample dict (new)

        render_image(sample["objects"], image_dir / img_filename, bg_color=bg)

        dataset.append(sample)
        idx += 1

        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{N_TRIPLE} done")

    # ----------------------------------------------------------
    # Split dataset 80% train / 10% val / 10% test
    # Note: random.shuffle is commented out to keep the original
    # ordering (single → binary → triple) across splits.
    # ----------------------------------------------------------
    random.shuffle(dataset)

    n_train = int(TOTAL * 0.8)   # 1600
    n_val   = int(TOTAL * 0.1)   # 200
    # n_test  = rest               # 200 (whatever remains)

    # Slice the list into three non-overlapping splits
    train_set = dataset[:n_train]
    val_set   = dataset[n_train:n_train + n_val]
    test_set  = dataset[n_train + n_val:]

    # Re-index sample IDs from 0 within each split so IDs are unique per file
    for i, s in enumerate(train_set):
        s["id"] = i
    for i, s in enumerate(val_set):
        s["id"] = i
    for i, s in enumerate(test_set):
        s["id"] = i

    # Save each split as a JSON file and print a type breakdown
    for split_name, split_data in [("train", train_set),
                                    ("val",   val_set),
                                    ("test",  test_set)]:
        json_path = output_dir / f"{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        # Count how many samples of each type ended up in this split
        type_counts = {}
        for s in split_data:
            type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
        print(f"\n  {split_name}.json : {len(split_data)} samples  {type_counts}")

    # Print a final summary of where everything was saved
    print(f"\n{'=' * 60}")
    print(f"Dataset saved to: {output_dir}")
    print(f"  train.json : {len(train_set)} samples (80%)")
    print(f"  val.json   : {len(val_set)} samples (10%)")
    print(f"  test.json  : {len(test_set)} samples (10%)")
    print(f"  images/    : {TOTAL} PNG files")

    # Print 3 random example sentences and image paths for each scene type
    print(f"\n===== Sample Examples =====")
    for t in ["single", "binary", "triple"]:
        samples_of_type = [s for s in dataset if s["type"] == t]
        examples = random.sample(samples_of_type, min(3, len(samples_of_type)))
        print(f"\n  [{t}]")
        for s in examples:
            print(f"    sentence: {s['sentence']}")
            print(f"    image:    {s['image']}")

    # Save a stats summary JSON recording key numbers about the dataset
    stats = {
        "total_samples":   TOTAL,
        "train_count":     len(train_set),
        "val_count":       len(val_set),
        "test_count":      len(test_set),
        "single_count":    N_SINGLE,
        "binary_count":    N_BINARY,
        "triple_count":    N_TRIPLE,
        "objects":         list(OBJECTS.keys()),
        "colors":          list(COLORS.keys()),
        "sizes":           list(SIZES.keys()),
        "directions":      list(DIRECTIONS.keys()),
        "screen_positions": list(SCREEN_POSITIONS.keys()),
        "triple_layouts":  list(BOTH_SIDES_LAYOUTS.keys()),
        "image_size_px":   IMG_SIZE,
        "bg_colors":       BG_COLORS,    # record the background palette used (new)
    }
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats saved: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()