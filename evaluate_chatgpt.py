"""
evaluate_chatgpt_simple.py

Sends images to GPT-4o and saves responses to a JSON file.

Usage:
    python evaluate_chatgpt_simple.py --data_dir ./dataset --api_key sk-proj-YOUR-KEY
    
    Or if you used pick_test_samples.py:
    python evaluate_chatgpt_simple.py --test_json ./llm_test_samples.json --data_dir ./dataset --api_key sk-proj-YOUR-KEY
"""

import argparse
import base64
import json
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT — tells GPT-4o exactly what format and vocabulary to use
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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def image_to_base64(image_path):
    """Convert image file to base64 string for the API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_gpt4o(client, image_path, retries=3):
    """Send one image to GPT-4o and return its text response."""
    image_b64 = image_to_base64(image_path)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content.strip().rstrip(".")

        except Exception as e:
            print(f"    API error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
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
                        help="Your OpenAI API key")
    parser.add_argument("--test_json", type=str, default=None,
                        help="Path to JSON file with test samples. "
                             "Defaults to data_dir/test.json")
    parser.add_argument("--output",    type=str, default="chatgpt_results.json",
                        help="Output file name (default: chatgpt_results.json)")
    parser.add_argument("--delay",     type=float, default=1.0,
                        help="Seconds to wait between API calls (default: 1.0)")
    args = parser.parse_args()

    # ── Import OpenAI ─────────────────────────────────────────────────────────
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed. Run:  pip install openai")
        return

    client   = OpenAI(api_key=args.api_key)
    data_dir = Path(args.data_dir)

    # ── Load test samples ─────────────────────────────────────────────────────
    # Use provided JSON or default to test.json
    test_json_path = Path(args.test_json) if args.test_json else data_dir / "test.json"

    if not test_json_path.exists():
        print(f"ERROR: {test_json_path} not found.")
        return

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total = len(test_data)
    print(f"Loaded {total} samples from {test_json_path}")
    print(f"Sending to GPT-4o...")
    print(f"Estimated cost: ~${total * 0.02:.2f} USD")
    print(f"Estimated time: ~{total * (args.delay + 3) / 60:.1f} minutes")
    print("-" * 60)

    # ── Run through each image ────────────────────────────────────────────────
    results = []

    for i, item in enumerate(test_data):
        image_path = data_dir / item["image"]

        print(f"[{i+1:3d}/{total}]  {Path(item['image']).name}", end="  →  ")

        if not image_path.exists():
            print("IMAGE NOT FOUND — skipping")
            continue

        # Call GPT-4o
        gpt_response = query_gpt4o(client, image_path)
        gpt_response = gpt_response.strip().lower()

        print(gpt_response)

        # Save result
        results.append({
            "image":        item["image"],
            "type":         item.get("type", "unknown"),
            "ground_truth": item["sentence"].strip().lower(),
            "gpt_response": gpt_response,
        })

        # Save progress every 10 images so you don't lose work if it crashes
        if (i + 1) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"  [Saved progress: {i+1}/{total}]")

        # Wait between calls to avoid rate limits
        if i < total - 1:
            time.sleep(args.delay)

    # ── Save final results ────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("-" * 60)
    print(f"Done. {len(results)} results saved to {args.output}")


if __name__ == "__main__":
    main()
