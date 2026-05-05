# EMATM0067 — Introduction to AI and Text Analytics | Team 10, Task 5
### Trisha Sharma (2825149) — University of Bristol

Sub-repository: Individual contribution to the group project.
Group repository: https://github.com/uob-aita-team10/team10-index

 ### Overview
 
This repository implements a CNN Encoder + Transformer Decoder architecture for image captioning on synthetic geometric scenes. It serves as the third independent experimental axis in the group project, testing encoder adaptation effects using a decoder architecture that relies on neither parallel classification heads nor a recurrent hidden state.
The primary finding is consistent with the broader group conclusion: backbone fine-tuning is the decisive factor in performance, not decoder sophistication. Fine-tuning the encoder improved exact match from 47% to 88% on 1,600 training samples.

### Dateset
 dataset.py: Extended version of the group dataset generation script.

The only change from the base script is the addition of randomised pastel backgrounds. A palette of 8 soft pastel colours is defined, and each generated image is assigned a randomly chosen background. This prevents the model from learning that the background is always white, which would not generalise, and increases visual variation across the 2,000 generated scenes.

Scene types generated:

500 single-object scenes — one shape at a named screen position
1,000 binary scenes — two shapes with a spatial relation (above, below, left of, etc.)
500 triple scenes — two shapes flanking a third (left-and-right / top-and-bottom)

## Models
Model_transformer_decoder.py: CNN encoder + Transformer decoder with the ResNet18 backbone frozen throughout training.

Transformer decoder:

Stack of Transformer decoder layers, each with three operations:
- Masked self-attention: each word attends directly to all previous words
- Cross-attention: each word attends to all 49 image patches
- Feed-forward network: processes the combined information

Positional encoding added to word embeddings (required since Transformer processes all words in parallel)
Generates captions word by word using greedy decoding at inference

## Results

Results on 1,600 training samples (2,000-sample base dataset, 80/10/10 split):

| Configuration | Backbone | Exact Match | Token Accuracy | Speed |
|---|---|---|---|---|
| Frozen backbone | frozen | 47.00% | 88.80% | ~10s/epoch |
| Unfrozen backbone | fine-tuned | 88.00% | 98.41% | ~85s/epoch |

**Key observation:** 
Under the frozen backbone, the model showed systematic colour confusion — most notably predicting "blue" in place of "cyan". This is consistent with ImageNet pretrained features lacking the specialised colour representations needed for flat synthetic geometry. After fine-tuning, these errors disappeared entirely.

## LLM comparison

## LLM Evaluation

### Step 1 — Select test images

```bash
python pick_test_samples.py
```

Randomly selects 30 samples from `dataset/test.json` and copies the images to `llm_test_images/`.
The sample count can be changed by editing `N_SAMPLES` in the script.

### Step 2 — Evaluate with GPT-4o

```bash
python evaluate_chatgpt.py \
  --data_dir ./dataset \
  --test_json ./llm_test_samples.json \
  --api_key YOUR_OPENAI_KEY
```

Saves results to `chatgpt_results.json`.
Requires an OpenAI account with API credits (~$0.02 per image).

### Step 3 — Evaluate with Gemini

```bash
python evaluate_gemini.py \
  --data_dir ./dataset \
  --test_json ./llm_test_samples.json \
  --api_key YOUR_GEMINI_KEY \
  --delay 10
```

Saves results to `gemini_results.json`.
Free — uses `gemini-2.0-flash` (1,500 requests/day limit). Use `--delay 10` to avoid rate limits.

> **Note:** If `llm_test_samples.json` is not present, both scripts fall back to `dataset/test.json`.

### LLM Prompt

Both scripts use the same structured prompt, specifying the exact vocabulary (sizes, colours, shapes,
relations) and sentence format required. This ensures outputs are comparable to ground truth using
exact match.

