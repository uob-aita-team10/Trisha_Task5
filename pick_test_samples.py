import json
import random
import shutil
from pathlib import Path

# Load your test set
with open("./dataset/test.json", "r") as f:
    test_data = json.load(f)

# Pick 6 random samples (or filter by type if you want)
random.seed(42)                        # same seed = same 6 images every time
samples = random.sample(test_data, 6   )

# Save the 6 samples to a small JSON for reference
with open("llm_test_samples.json", "w") as f:
    json.dump(samples, f, indent=2)

# Copy the 6 images into a separate folder so you can easily upload them
output_dir = Path("llm_test_images")
output_dir.mkdir(exist_ok=True)

for item in samples:
    src = Path("./dataset") / item["image"]
    dst = output_dir / Path(item["image"]).name
    shutil.copy(src, dst)

print(f"Saved {len(samples)} samples to llm_test_samples.json")
print(f"Copied {len(samples)} images to llm_test_images/")
print("\nFirst 3 samples:")
for s in samples[:3]:
    print(f"  {s['image']}  →  {s['sentence']}")