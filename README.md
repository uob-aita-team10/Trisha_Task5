 # CNN + Transformer Image Captioning
 
 A coursework project (EMATM0067) that trains a CNN encoder + Transformer decoder to generate natural language 
 descriptions of synthetic geometric scenes. Also includes zero-shot LLM baselines using GPT-4o and Gemini.

### Dateset
 dataset_ts.py: The same dataset and code, only added a section to add colour in the background different than while.

## Models
Model_transformer_decoder.py: CNN encoder + Transformer decoder with the ResNet18 backbone frozen throughout training.

Transformer decoder: generates captions word by word, At each step the decoder uses self-attention to look at all 
previously generated words directly, and cross-attention to focus on the relevant image patches

Model_transformer_decoder_backbone.py: With Backbone Fine-Tuning

#### results for 1,600 training samples
Frozen backbone : 47.00% exact match | 88.80% token accuracy – quick but confused visually similar colours | 30 epochs

Unfrozen backbone: 88.00% exact match | 98.41% token accuracy – very slow but accurate | 50 epochs


## LLM comparison

pick_test_samples.py: Randomly selects 6(could be changed) samples from dataset/test.json and copies the images to llm_test_images/ for easy upload.

**GPT-4o (evaluate_chatgpt.py):** Saves results to chatgpt_results.json
```bash
python evaluate_chatgpt_simple.py \
  --data_dir ./dataset \
  --test_json ./llm_test_samples.json \
  --api_key YOUR_OPENAI_KEY
```

**Gemini 2.0 Flash (evaluate_gemini.py):** Saves results to gemini_results.json
```bash
python evaluate_gemini.py \
  --data_dir ./dataset \
  --test_json ./llm_test_samples.json \
  --api_key YOUR_GEMINI_KEY \
  --delay 10
```

**Note**
evaluate_chatgpt.py and evaluate_gemini.py uses llm_test_images (created using pick_test_samples.py) for evaluation. In case llm_test_images is not present, it takes dataset/test.json as a default option
