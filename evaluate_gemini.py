"""
evaluate_gemini.py

Sends images to Google Gemini and saves responses to a JSON file.

FREE — uses gemini-2.0-flash at no cost.

Setup:
    1. Get free API key from: https://aistudio.google.com  (no credit card needed)
    2. pip install google-genai Pillow

Usage:
    python evaluate_gemini.py --data_dir ./dataset --api_key AIzaSy-YOUR-KEY --test_json ./llm_test_samples.json --delay 10

Output:
    gemini_results.json  — contains every image path, ground truth, and Gemini's response

Note on delay:
    Use --delay 10 to avoid hitting the per-minute rate limit.
    If you still get 429 errors, your daily quota is used up — wait until tomorrow.
"""

import argparse
import json
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT — tells Gemini exactly what format and vocabulary to use
# ─────────────────────────────────────────────────────────────────────────────

PROMPT = """Look at this image carefully. It contains simple geometric shapes on a white background.

Describe the image in ONE sentence using ONLY the words from the lists below.
Do not use any other words. Do not add any explanation.

SIZES (choose one): small, medium, big
COLOURS (choose one): red, blue, green, yellow, purple, orange, pink, cyan
SHAPES (choose one): sphere, cube, triangle, diamond, star, pentagon, hexagon, cross

If the image has ONE object, use this format exactly:
"a [size] [colour] [shape] appears at the [position] of the image"
Where position is one of: center, top, bottom, left, right, top left, top right, bottom left, bottom right

If the image has TWO objects, use this format exactly:
"a [size] [colour] [shape] is [relation] a [size] [colour] [shape]"
Where relation is one of: above, below, to the left of, to the right of, upper left of, upper right of, lower left of, lower right of

If the image has THREE objects (two flanking one in the middle), use this format exactly:
"a [size] [colour] [shape] and a [size] [colour] [shape] are on the [side] of a [size] [colour] [shape]"
Where side is one of: left and right, top and bottom

Output ONLY the sentence. Nothing else."""


# ─────────────────────────────────────────────────────────────────────────────
# QUERY FUNCTION — uses new google-genai library (no deprecation warning)
# ─────────────────────────────────────────────────────────────────────────────

def query_gemini(client, image_path, retries=3):
    """
    Send one image to Gemini and return its text response.
    Uses the new google.genai library instead of deprecated google.generativeai.
    Retries up to 3 times on network or API errors.
    """
    from PIL import Image

    # Open image with PIL — the new genai library accepts PIL images directly
    image = Image.open(image_path).convert("RGB")

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model    = "gemini-2.0-flash",
                contents = [PROMPT, image],
            )
            # Clean up: lowercase and remove any trailing punctuation
            return response.text.strip().lower().rstrip(".")

        except Exception as e:
            print(f"    API error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)   # wait 5 seconds before retrying
            else:
                return "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, required=True,
                        help="Path to dataset folder containing images/")
    parser.add_argument("--api_key",   type=str, required=True,
                        help="Your Gemini API key from aistudio.google.com")
    parser.add_argument("--test_json", type=str, default=None,
                        help="Path to your test samples JSON file. "
                             "Defaults to data_dir/test.json")
    parser.add_argument("--output",    type=str, default="gemini_results.json",
                        help="Output file name (default: gemini_results.json)")
    parser.add_argument("--delay",     type=float, default=10.0,
                        help="Seconds to wait between API calls (default: 10.0). "
                             "Increase if you keep hitting rate limits.")
    args = parser.parse_args()

    # ── Import new google-genai library (replaces deprecated google-generativeai)
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed.")
        print("Run:  pip install google-genai")
        return

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed.")
        print("Run:  pip install Pillow")
        return

    # ── Setup Gemini client ───────────────────────────────────────────────────
    # New library uses Client object instead of genai.configure()
    # This is what removes the FutureWarning you were seeing
    client = genai.Client(api_key=args.api_key)

    data_dir = Path(args.data_dir)

    # ── Load test samples ─────────────────────────────────────────────────────
    test_json_path = Path(args.test_json) if args.test_json else data_dir / "test.json"

    if not test_json_path.exists():
        print(f"ERROR: {test_json_path} not found.")
        return

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total = len(test_data)
    print(f"Loaded {total} samples from {test_json_path}")
    print(f"Model: gemini-2.0-flash (FREE)")
    print(f"Delay between calls: {args.delay}s")
    print(f"Estimated time: ~{total * (args.delay + 2) / 60:.1f} minutes")
    print("-" * 60)

    # ── Run through each image ────────────────────────────────────────────────
    results = []

    for i, item in enumerate(test_data):
        image_path = data_dir / item["image"]

        print(f"[{i+1:3d}/{total}]  {Path(item['image']).name}", end="  →  ")

        if not image_path.exists():
            print("IMAGE NOT FOUND — skipping")
            continue

        # Query Gemini
        gemini_response = query_gemini(client, image_path)

        print(gemini_response)

        # Save result
        results.append({
            "image":           item["image"],
            "type":            item.get("type", "unknown"),
            "ground_truth":    item["sentence"].strip().lower(),
            "gemini_response": gemini_response,
        })

        # Save progress every 10 images so you don't lose work if it crashes
        if (i + 1) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"  [Saved progress: {i+1}/{total}]")

        # Wait between calls to stay within rate limits
        if i < total - 1:
            time.sleep(args.delay)

    # ── Save final results ────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("-" * 60)
    print(f"Done. {len(results)} results saved to {args.output}")


if __name__ == "__main__":
    main()