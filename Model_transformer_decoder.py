"""

CNN Encoder + Transformer Decoder — OPTIMISED VERSION

Usage (CPU — fast settings):
    python Model_transformer_decoder.py --data_dir ./dataset --epochs 30

Usage (GPU — full settings):
    python Model_transformer_decoder.py --data_dir ./dataset --epochs 30 --embed_dim 256 --num_heads 8 --num_layers 3 --ff_dim 512
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms


# ─────────────────────────────────────────────────────────────────────────────
# SPECIAL TOKENS
# ─────────────────────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """Converts words <-> integer indices."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        for i, tok in enumerate([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]):
            self.word2idx[tok] = i
            self.idx2word[i]   = tok

    def build(self, sentences, min_freq=1):
        for sent in sentences:
            self.word_freq.update(sent.lower().split())
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
                idx += 1

    def encode(self, sentence, max_len):
        indices = [SOS_IDX]
        for token in sentence.lower().split():
            indices.append(self.word2idx.get(token, UNK_IDX))
        indices.append(EOS_IDX)
        if len(indices) > max_len:
            indices = indices[:max_len]
        while len(indices) < max_len:
            indices.append(PAD_IDX)
        return indices

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            words.append(self.idx2word.get(idx, UNK_TOKEN))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (used only for the pre-caching step)
# ─────────────────────────────────────────────────────────────────────────────

class ShapeCaptionDataset(Dataset):
    """Loads raw image-caption pairs from JSON. Used only during pre-caching."""

    def __init__(self, json_path, data_dir, vocab, max_len, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_dir  = Path(data_dir)
        self.vocab     = vocab
        self.max_len   = max_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item    = self.data[index]
        image   = Image.open(self.data_dir / item["image"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = torch.tensor(
            self.vocab.encode(item["sentence"], self.max_len),
            dtype=torch.long
        )
        return image, caption


# ─────────────────────────────────────────────────────────────────────────────
# SPEEDUP 1 — PRE-CACHE IMAGE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def precompute_features(dataset, backbone, device, batch_size=64):
    """
    Run ALL images through the frozen CNN backbone ONCE before training starts.
    Save the resulting feature tensors in memory.

    WHY THIS IS FAST:
        Normally the backbone runs on every image at every epoch.
        30 epochs × 1600 images = 48,000 backbone forward passes.

        Pre-caching runs the backbone ONCE: 1600 passes total.
        That's a 30x reduction in backbone computation.

        The backbone is the slowest part on CPU (11M frozen parameters).
        After caching, each training step only runs the small Transformer
        decoder — much faster.

    Returns a TensorDataset of (image_features, captions) ready for training.
    """
    print("  Pre-computing image features (runs backbone once for all images)...")

    # Temporary loader — no shuffle needed, just reading in order
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []   # will collect (B, 49, embed_dim) tensors
    all_captions = []   # will collect (B, max_len) tensors

    backbone.eval()     # disable dropout in backbone
    with torch.no_grad():
        for i, (images, captions) in enumerate(loader):
            images = images.to(device)

            # Run backbone: (B, 3, 224, 224) → (B, 512, 7, 7)
            feats = backbone(images)             # (B, 512, 7, 7)

            # Move features back to CPU for storage (saves GPU memory)
            all_features.append(feats.cpu())
            all_captions.append(captions)

            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"    Processed {(i+1)*batch_size} / {len(dataset)} images...", end="\r")

    # Concatenate all batches into single tensors
    all_features = torch.cat(all_features, dim=0)   # (N, 512, 7, 7)
    all_captions = torch.cat(all_captions, dim=0)   # (N, max_len)

    print(f"\n  Done. Cached {len(all_features)} feature tensors.")
    print(f"  Feature tensor size: {all_features.shape}")

    # Return as TensorDataset — super fast to load during training
    # (everything is already in memory as tensors, no file I/O needed)
    return TensorDataset(all_features, all_captions)


# ─────────────────────────────────────────────────────────────────────────────
# CNN ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    ResNet18 encoder split into two parts:

    1. backbone  — the heavy pretrained CNN (frozen, used only for pre-caching)
    2. projection — a small trainable Linear layer (used every training step)

    During training, only the projection runs.
    The backbone runs once during pre-caching, then is never called again.
    """

    def __init__(self, embed_dim=128, pretrained=True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Remove last 2 layers to keep spatial 7x7 feature maps
        # [-1] = FC classifier, [-2] = global average pool
        modules        = list(resnet.children())[:-2]
        self.backbone  = nn.Sequential(*modules)   # outputs (B, 512, 7, 7)

        # Freeze backbone — we never want to update these weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Small trainable projection: 512 → embed_dim (applied to each of 49 patches)
        self.projection = nn.Linear(512, embed_dim)
        self.norm       = nn.LayerNorm(embed_dim)

    def project(self, raw_features):
        """
        Takes pre-cached raw backbone features and applies the projection.
        This is what runs every training step (fast — just a Linear layer).

        raw_features: (B, 512, 7, 7)  — cached backbone output
        returns:      (B, 49, embed_dim)
        """
        B, C, H, W = raw_features.shape           # B, 512, 7, 7

        # Rearrange spatial grid: (B, 512, 7, 7) → (B, 49, 512)
        features = raw_features.permute(0, 2, 3, 1)   # (B, 7, 7, 512)
        features = features.reshape(B, H * W, C)       # (B, 49, 512)

        # Project: 512 → embed_dim
        features = self.projection(features)            # (B, 49, embed_dim)
        features = self.norm(features)                  # normalise

        return features

    def forward(self, images):
        """
        Full forward pass (backbone + projection).
        Only used during inference — training uses project() with cached features.
        """
        with torch.no_grad():
            raw = self.backbone(images)   # (B, 512, 7, 7)
        return self.project(raw)


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Adds position information to word embeddings using sine/cosine waves.
    Needed because Transformer processes all words at once with no built-in
    sense of order (unlike LSTM which processes left to right).
    """

    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions: sine
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions:  cosine
        pe = pe.unsqueeze(0)                            # add batch dimension
        self.register_buffer("pe", pe)                  # not a trainable parameter

    def forward(self, x):
        # x: (B, seq_len, embed_dim)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER DECODER
# ─────────────────────────────────────────────────────────────────────────────

class TransformerDecoder(nn.Module):
    """
    Generates captions using a stack of Transformer decoder layers.

    Each layer has three operations:
        1. Masked self-attention   — each word looks at all previous words
        2. Cross-attention         — each word looks at all 49 image patches
        3. Feed-forward network    — processes the combined information

    Default settings are reduced for CPU speed:
        embed_dim=128, num_heads=4, num_layers=2, ff_dim=256
    """

    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=2, ff_dim=256, max_len=25, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # Word index → dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # Add position information to word vectors
        self.pos_encoding = PositionalEncoding(embed_dim, max_len + 10, dropout)

        # Stack of Transformer decoder layers
        # Each layer: self-attention + cross-attention + feed-forward
        decoder_layer = nn.TransformerDecoderLayer(
            d_model        = embed_dim,    # vector size at each position
            nhead          = num_heads,    # parallel attention heads
            dim_feedforward= ff_dim,       # internal FF hidden size
            dropout        = dropout,
            batch_first    = True,         # (batch, seq, features) ordering
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final projection: embed_dim → vocab_size (one score per word)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.max_len = max_len

    def make_causal_mask(self, seq_len, device):
        """
        Upper triangular mask — prevents attending to future tokens.
        Word at position t can only see positions 0..t.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self, image_features, captions):
        """
        Training forward pass with teacher forcing.

        image_features: (B, 49, embed_dim)  — from encoder
        captions:       (B, max_len)         — ground truth tokens

        Returns logits: (B, max_len-1, vocab_size)
        """
        # Input: all tokens except the last  [SOS, w1, w2, ..., wN]
        # Target (computed in training loop): all tokens except first [w1, w2, ..., EOS]
        tgt     = captions[:, :-1]       # (B, max_len-1)
        seq_len = tgt.size(1)

        # 1. Embed tokens and scale (standard Transformer trick for stability)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_dim)  # (B, seq_len, embed_dim)

        # 2. Add positional encoding
        tgt_emb = self.pos_encoding(tgt_emb)                        # (B, seq_len, embed_dim)

        # 3. Causal mask — hide future words during training
        causal_mask = self.make_causal_mask(seq_len, tgt.device)

        # 4. Padding mask — don't attend to PAD tokens
        tgt_key_padding_mask = (tgt == PAD_IDX)                     # (B, seq_len)

        # 5. Run through Transformer decoder layers
        #    tgt = word sequence being built
        #    memory = image patches to attend to via cross-attention
        output = self.transformer(
            tgt                  = tgt_emb,
            memory               = image_features,
            tgt_mask             = causal_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
        )                                                            # (B, seq_len, embed_dim)

        # 6. Project to vocabulary scores
        logits = self.fc_out(output)                                 # (B, seq_len, vocab_size)

        return logits


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CNNTransformerCaptioner(nn.Module):
    """Full model: CNN Encoder + Transformer Decoder."""

    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=2, ff_dim=256, max_len=25,
                 dropout=0.1, pretrained=True):
        super().__init__()

        self.encoder = CNNEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.decoder = TransformerDecoder(
            vocab_size = vocab_size,
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            num_layers = num_layers,
            ff_dim     = ff_dim,
            max_len    = max_len,
            dropout    = dropout,
        )
        self.max_len = max_len

    def forward_cached(self, raw_features, captions):
        """
        Training forward pass using PRE-CACHED backbone features.
        Skips the slow backbone — only runs projection + Transformer.

        raw_features: (B, 512, 7, 7)  — cached from pre-caching step
        captions:     (B, max_len)
        """
        # Apply projection to convert raw features to embed_dim
        image_features = self.encoder.project(raw_features.to(next(self.encoder.projection.parameters()).device))
        # Decode
        logits = self.decoder(image_features, captions)
        return logits

    def forward(self, images, captions):
        """
        Standard forward pass — runs full backbone + decoder.
        Used during inference (greedy_decode).
        """
        image_features = self.encoder(images)
        logits         = self.decoder(image_features, captions)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY DECODING  (FIXED — no more IndexError)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_decode(model, images, max_len, device):
    """
    Generate captions one word at a time at inference.

    FIX vs original:
        The original called model.decoder(image_features, generated)
        which internally did captions[:, :-1], turning a 1-token sequence
        into a 0-token sequence → IndexError on dimension 1 with size 0.

        This fixed version bypasses the training forward() and calls the
        Transformer's internal components directly with no slicing.
    """
    model.eval()
    B = images.size(0)

    with torch.no_grad():
        # Encode images → spatial patch features
        image_features = model.encoder(images)   # (B, 49, embed_dim)

        # Start with <SOS> token for every image in the batch
        generated = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            seq_len = generated.size(1)

            # ── Embed current generated sequence ────────────────────────────
            tgt_emb = model.decoder.embedding(generated) * math.sqrt(model.decoder.embed_dim)
            tgt_emb = model.decoder.pos_encoding(tgt_emb)    # (B, seq_len, embed_dim)

            # ── Causal mask to prevent peeking at future positions ──────────
            causal_mask = model.decoder.make_causal_mask(seq_len, device)

            # ── Run through Transformer decoder layers ──────────────────────
            output = model.decoder.transformer(
                tgt    = tgt_emb,
                memory = image_features,
                tgt_mask = causal_mask,
            )                                                 # (B, seq_len, embed_dim)

            # ── Take only the LAST position (most recent prediction) ────────
            last_output = output[:, -1, :]                   # (B, embed_dim)
            next_logits = model.decoder.fc_out(last_output)  # (B, vocab_size)
            next_token  = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # ── Append predicted token to sequence ──────────────────────────
            generated = torch.cat([generated, next_token], dim=1)   # (B, seq_len+1)

            # ── Early stop if all sequences have generated EOS ───────────────
            if (next_token == EOS_IDX).all():
                break

    return generated   # (B, generated_length)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS  (use cached features for speed)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One training epoch using PRE-CACHED features.
    The loader yields (raw_features, captions) — no images, no backbone.
    """
    model.train()
    total_loss = 0.0
    n          = 0

    for raw_features, captions in loader:
        # Move to device
        raw_features = raw_features.to(device)
        captions     = captions.to(device)

        optimizer.zero_grad()

        # Forward pass using cached features (fast — no backbone)
        logits = model.forward_cached(raw_features, captions)
        # logits: (B, max_len-1, vocab_size)

        # Target: caption shifted left by 1
        # Input:  [SOS, w1, w2, w3]
        # Target: [w1,  w2, w3, EOS]
        targets = captions[:, 1:]   # (B, max_len-1)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),   # (B*(max_len-1), vocab_size)
            targets.reshape(-1)                    # (B*(max_len-1),)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size   = raw_features.size(0)
        total_loss  += loss.item() * batch_size
        n           += batch_size

    return total_loss / n


def evaluate_loss(model, loader, criterion, device):
    """Validation loss using pre-cached features."""
    model.eval()
    total_loss = 0.0
    n          = 0

    with torch.no_grad():
        for raw_features, captions in loader:
            raw_features = raw_features.to(device)
            captions     = captions.to(device)

            logits   = model.forward_cached(raw_features, captions)
            targets  = captions[:, 1:]

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            batch_size   = raw_features.size(0)
            total_loss  += loss.item() * batch_size
            n           += batch_size

    return total_loss / n


def compute_accuracy(model, loader, vocab, max_len, device):
    """
    Accuracy on a dataset split.
    loader here yields raw images (not cached features) for greedy_decode.
    """
    model.eval()
    exact_match   = 0
    token_correct = 0
    token_total   = 0
    total         = 0
    examples      = []

    with torch.no_grad():
        for images, captions in loader:
            images   = images.to(device)
            captions = captions.to(device)

            # Use greedy decode (runs full encoder internally)
            preds = greedy_decode(model, images, max_len, device)   # (B, gen_len)

            for i in range(images.size(0)):
                pred_sent = vocab.decode(preds[i].tolist())
                gt_sent   = vocab.decode(captions[i].tolist())

                if pred_sent == gt_sent:
                    exact_match += 1

                pred_tokens = pred_sent.split()
                gt_tokens   = gt_sent.split()
                max_tok_len = max(len(pred_tokens), len(gt_tokens))

                for j in range(max_tok_len):
                    pt = pred_tokens[j] if j < len(pred_tokens) else ""
                    gt = gt_tokens[j]   if j < len(gt_tokens)   else ""
                    if pt == gt:
                        token_correct += 1
                    token_total += 1

                total += 1
                if len(examples) < 10:
                    examples.append((gt_sent, pred_sent))

    exact_acc = exact_match   / total        if total       > 0 else 0
    token_acc = token_correct / token_total  if token_total > 0 else 0
    return exact_acc, token_acc, examples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CNN + Transformer Image Captioning (Fast)")

    parser.add_argument("--data_dir",   type=str,   default="./dataset")
    # ── Smaller defaults for CPU speed ──────────────────────────────────────
    parser.add_argument("--embed_dim",  type=int,   default=128,
                        help="Embedding size. Use 256 on GPU for higher accuracy.")
    parser.add_argument("--num_heads",  type=int,   default=4,
                        help="Attention heads. Must divide embed_dim evenly.")
    parser.add_argument("--num_layers", type=int,   default=2,
                        help="Transformer decoder layers. Use 3-4 on GPU.")
    parser.add_argument("--ff_dim",     type=int,   default=256,
                        help="Feed-forward hidden size. Use 512 on GPU.")
    parser.add_argument("--max_len",    type=int,   default=25)
    parser.add_argument("--batch_size", type=int,   default=64,
                        help="Larger batch = fewer steps per epoch = faster.")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--save_dir",   type=str,   default="./checkpoints_transformer_fast")

    args = parser.parse_args()

    # ── Device setup ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pin_memory only helps on GPU — disable on CPU to remove the warning
    use_pin_memory = (device.type == "cuda")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  Running on CPU. Consider Google Colab (free GPU) for ~15x speedup.")
        print("  tip: runtime → change runtime type → T4 GPU")

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Vocabulary ───────────────────────────────────────────────────
    print("\n[Step 1] Building vocabulary...")
    with open(data_dir / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_sentences = [item["sentence"] for item in train_data]
    vocab = Vocabulary()
    vocab.build(train_sentences)
    vocab_size = len(vocab)

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max caption length: {args.max_len}")

    vocab_path = save_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "word2idx": vocab.word2idx,
            "idx2word": {str(k): v for k, v in vocab.idx2word.items()}
        }, f, indent=2)
    print(f"  Vocabulary saved: {vocab_path}")

    # ── Step 2: Image transforms ─────────────────────────────────────────────
    print("\n[Step 2] Preparing datasets...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Raw datasets — used for pre-caching and final evaluation
    train_dataset = ShapeCaptionDataset(data_dir / "train.json", data_dir, vocab, args.max_len, transform)
    val_dataset   = ShapeCaptionDataset(data_dir / "val.json",   data_dir, vocab, args.max_len, transform)
    test_dataset  = ShapeCaptionDataset(data_dir / "test.json",  data_dir, vocab, args.max_len, transform)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Step 3: Build model ──────────────────────────────────────────────────
    print(f"\n[Step 3] Building CNN + Transformer model...")
    print(f"  embed_dim={args.embed_dim}, num_heads={args.num_heads}, "
          f"num_layers={args.num_layers}, ff_dim={args.ff_dim}")

    model = CNNTransformerCaptioner(
        vocab_size = vocab_size,
        embed_dim  = args.embed_dim,
        num_heads  = args.num_heads,
        num_layers = args.num_layers,
        ff_dim     = args.ff_dim,
        max_len    = args.max_len,
        dropout    = args.dropout,
        pretrained = True,
    ).to(device)

    # ── SPEEDUP: torch.compile (PyTorch 2.0+, free ~10-30% speedup) ─────────
    # Compiles the model's computation graph for faster execution.
    # Falls back silently if not supported (older PyTorch versions).
    try:
        model = torch.compile(model)
        print("  torch.compile() applied — extra speed boost active.")
    except Exception:
        pass   # older PyTorch — just skip, no problem

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Step 4: Pre-cache image features ─────────────────────────────────────
    print("\n[Step 4] Pre-caching image features (the big speedup)...")
    print("  This runs the backbone ONCE for all images.")
    print("  Training will then skip the backbone entirely each epoch.")

    # Use the encoder's backbone for caching (move to device for speed)
    # Access underlying model if torch.compile wraps it
    try:
        backbone = model._orig_mod.encoder.backbone
    except AttributeError:
        backbone = model.encoder.backbone
    backbone = backbone.to(device)

    t_cache = time.time()
    train_cached = precompute_features(train_dataset, backbone, device, batch_size=args.batch_size)
    val_cached   = precompute_features(val_dataset,   backbone, device, batch_size=args.batch_size)
    print(f"  Pre-caching done in {time.time() - t_cache:.1f}s")

    # Fast loaders from cached TensorDatasets
    # (no file I/O, no image decoding — just tensor slicing)
    train_loader = DataLoader(train_cached, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_cached,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    # Test loader uses original dataset (needs full forward pass for greedy decode)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    # ── Step 5: Loss and Optimiser ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Step 6: Training loop ────────────────────────────────────────────────
    print(f"\n[Step 5] Training for {args.epochs} epochs...")
    print("-" * 70)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate_loss(model,   val_loader,   criterion,           device)

        elapsed = time.time() - t0
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {lr_now:.6f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / "best_transformer_fast.pth"
            # Save the original model (not the compiled wrapper)
            try:
                state_dict = model._orig_mod.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save({
                "epoch":            epoch,
                "model_state_dict": state_dict,
                "val_loss":         val_loss,
                "embed_dim":        args.embed_dim,
                "num_heads":        args.num_heads,
                "num_layers":       args.num_layers,
                "ff_dim":           args.ff_dim,
                "max_len":          args.max_len,
                "vocab_size":       vocab_size,
            }, ckpt_path)
            print(f"  ✓ Best model saved: {ckpt_path}")

    # ── Step 7: Evaluate on test set ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("[Step 6] Evaluating on test set...")

    ckpt_path = save_dir / "best_transformer_fast.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            model._orig_mod.load_state_dict(ckpt["model_state_dict"])
        except AttributeError:
            model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    exact_acc, token_acc, examples = compute_accuracy(
        model, test_loader, vocab, args.max_len, device
    )

    print(f"\n  Exact match accuracy: {exact_acc:.4f}  ({exact_acc * 100:.2f}%)")
    print(f"  Token-level accuracy: {token_acc:.4f}  ({token_acc * 100:.2f}%)")

    print(f"\n  Sample predictions:")
    for gt, pred in examples[:5]:
        print(f"    GT:   {gt}")
        print(f"    Pred: {pred}")
        print()

    results = {
        "model":            "CNN + Transformer (Fast)",
        "embed_dim":        args.embed_dim,
        "num_heads":        args.num_heads,
        "num_layers":       args.num_layers,
        "ff_dim":           args.ff_dim,
        "epochs":           args.epochs,
        "best_val_loss":    best_val_loss,
        "exact_match_acc":  exact_acc,
        "token_level_acc":  token_acc,
    }
    results_path = save_dir / "results_transformer_fast.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {results_path}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()