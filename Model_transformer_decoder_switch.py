"""
Model_transformer_decoder_switch.py

CNN Encoder + Transformer Decoder — with backbone fine-tuning support.

Two modes:
    FROZEN backbone  (default, fast):
        python Model_transformer_decoder_backbone.py --data_dir ./dataset --epochs 50
        → backbone cached once, ~10s/epoch, ~47% exact match

    UNFROZEN backbone (fine-tuning, slower but more accurate):
        python Model_transformer_decoder_backbone.py --data_dir ./dataset --epochs 50 --unfreeze_backbone
        → backbone runs every step, ~45s/epoch, ~80-99% exact match

Changes:
    1. unfreeze_backbone flag added
    2. CNNEncoder accepts freeze_backbone argument
    3. Separate learning rates: backbone 10x lower than decoder
    4. Caching automatically disabled when backbone is unfrozen
    5. train_one_epoch_full / evaluate_loss_full added for unfrozen mode
    6. greedy_decode updated to handle unfrozen backbone correctly

Usage:
    Frozen backbone (fast, ~10s/epoch):
        python Model_transformer_decoder_switch.py --data_dir ./dataset --epochs 50

    Unfrozen backbone (fine-tuning, slower but more accurate):
        python Model_transformer_decoder_switch.py --data_dir ./dataset --epochs 50 --unfreeze_backbone

    Full model settings (used for final results):
        python Model_transformer_decoder_switch.py --data_dir ./dataset --epochs 50
            --embed_dim 256 --num_heads 8 --num_layers 3 --ff_dim 512 --unfreeze_backbone

Output:
    checkpoints_transformer_final/best_transformer_frozen.pth   (frozen run)
    checkpoints_transformer_final/best_transformer_unfrozen.pth (unfrozen run)
    checkpoints_transformer_final/results_frozen.json
    checkpoints_transformer_final/results_unfrozen.json
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
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ShapeCaptionDataset(Dataset):
   
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
# PRE-CACHE IMAGE FEATURES  (used only when backbone is FROZEN)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_features(dataset, backbone, device, batch_size=64):
    print("  Pre-computing image features (runs backbone once for all images)...")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_captions = []

    backbone.eval()
    with torch.no_grad():
        for i, (images, captions) in enumerate(loader):
            images = images.to(device)
            feats  = backbone(images)          # (B, 512, 7, 7)
            all_features.append(feats.cpu())   # store on CPU to save GPU memory
            all_captions.append(captions)

            if (i + 1) % 5 == 0:
                print(f"    Processed {min((i+1)*batch_size, len(dataset))} "
                      f"/ {len(dataset)} images...", end="\r")

    all_features = torch.cat(all_features, dim=0)   # (N, 512, 7, 7)
    all_captions = torch.cat(all_captions, dim=0)   # (N, max_len)

    print(f"\n  Done. Cached {len(all_features)} feature tensors.")
    print(f"  Feature tensor size: {all_features.shape}")

    return TensorDataset(all_features, all_captions)


# ─────────────────────────────────────────────────────────────────────────────
# CNN ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    ResNet18 image encoder that outputs 49 spatial patch features (7x7 grid).

    CHANGE: now accepts freeze_backbone argument.
        freeze_backbone=True  → backbone weights locked, only projection trains
                                 caching is possible (fast ~10s/epoch)
        freeze_backbone=False → all backbone weights update during training
                                 caching NOT possible (slower ~45s/epoch)
                                 but backbone adapts to flat-colour images
                                 → much higher accuracy
    """

    def __init__(self, embed_dim=128, pretrained=True, freeze_backbone=True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        
        modules       = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)   

        
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

        self.freeze_backbone = freeze_backbone

        
        self.projection = nn.Linear(512, embed_dim)
        self.norm       = nn.LayerNorm(embed_dim)

    def project(self, raw_features):
        
        B, C, H, W = raw_features.shape

        
        features = raw_features.permute(0, 2, 3, 1)  
        features = features.reshape(B, H * W, C)       

        
        features = self.projection(features)            
        features = self.norm(features)

        return features

    def forward(self, images):
        """
        Full forward pass through backbone + projection.

        When frozen:   backbone wrapped in no_grad (no gradients computed)
        When unfrozen: backbone runs normally (gradients flow through it)
        """
        if self.freeze_backbone:
            # Frozen: no need to compute gradients through backbone
            with torch.no_grad():
                raw = self.backbone(images)   
        else:
            # Unfrozen: gradients flow through backbone for fine-tuning
            raw = self.backbone(images)       

        return self.project(raw)             


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
   
    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims: sine wave
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims:  cosine wave
        pe = pe.unsqueeze(0)                            # add batch dimension
        self.register_buffer("pe", pe)                  # fixed, not trained

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER DECODER
# ─────────────────────────────────────────────────────────────────────────────

class TransformerDecoder(nn.Module):
    """
    Generates captions using stacked Transformer decoder layers.

    Each layer has 3 operations:
        1. Masked self-attention  — each word looks at all PREVIOUS words directly
        2. Cross-attention        — each word looks at all 49 image patches
        3. Feed-forward network   — processes the combined information

    """

    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=2, ff_dim=256, max_len=25, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        
        self.embedding    = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        
        self.pos_encoding = PositionalEncoding(embed_dim, max_len + 10, dropout)

       
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = embed_dim,   
            nhead           = num_heads,   
            dim_feedforward = ff_dim,      
            dropout         = dropout,
            batch_first     = True,        
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

       
        self.fc_out  = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def make_causal_mask(self, seq_len, device):
        
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self, image_features, captions):
        
        tgt     = captions[:, :-1]    
        seq_len = tgt.size(1)

        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_dim)

       
        tgt_emb = self.pos_encoding(tgt_emb)

       
        causal_mask = self.make_causal_mask(seq_len, tgt.device)

       
        tgt_key_padding_mask = (tgt == PAD_IDX)

        
        output = self.transformer(
            tgt                  = tgt_emb,
            memory               = image_features,    
            tgt_mask             = causal_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
        )                                             
       
        return self.fc_out(output)                     


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CNNTransformerCaptioner(nn.Module):
    

    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=2, ff_dim=256, max_len=25,
                 dropout=0.1, pretrained=True, freeze_backbone=True):
        super().__init__()

        
        self.encoder = CNNEncoder(
            embed_dim       = embed_dim,
            pretrained      = pretrained,
            freeze_backbone = freeze_backbone, 
        )
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
        
        
        proj_device    = next(self.encoder.projection.parameters()).device
        image_features = self.encoder.project(raw_features.to(proj_device))
        return self.decoder(image_features, captions)

    def forward(self, images, captions):
        
        image_features = self.encoder(images)
        return self.decoder(image_features, captions)


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY DECODING  (inference — no teacher forcing)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_decode(model, images, max_len, device):
    
    model.eval()
    B = images.size(0)

    with torch.no_grad():
        
        image_features = model.encoder(images)  

        
        generated = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            seq_len = generated.size(1)

           
            tgt_emb = model.decoder.embedding(generated) * math.sqrt(model.decoder.embed_dim)
            tgt_emb = model.decoder.pos_encoding(tgt_emb)

            
            causal_mask = model.decoder.make_causal_mask(seq_len, device)

            
            output = model.decoder.transformer(
                tgt      = tgt_emb,
                memory   = image_features,
                tgt_mask = causal_mask,
            )                                              

            
            last_output = output[:, -1, :]              
            next_logits = model.decoder.fc_out(last_output) 
            next_token  = next_logits.argmax(dim=-1, keepdim=True)  

            
            generated = torch.cat([generated, next_token], dim=1)

           
            if (next_token == EOS_IDX).all():
                break

    return generated  


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS — FROZEN MODE (uses cached features, fast)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch_cached(model, loader, criterion, optimizer, device):
    
    model.train()
    total_loss = 0.0
    n          = 0

    for raw_features, captions in loader:
        raw_features = raw_features.to(device)
        captions     = captions.to(device)

        optimizer.zero_grad()

        
        logits  = model.forward_cached(raw_features, captions)
        targets = captions[:, 1:]   

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * raw_features.size(0)
        n          += raw_features.size(0)

    return total_loss / n


def evaluate_loss_cached(model, loader, criterion, device):
    """Validation loss with cached features — frozen backbone mode."""
    model.eval()
    total_loss = 0.0
    n          = 0

    with torch.no_grad():
        for raw_features, captions in loader:
            raw_features = raw_features.to(device)
            captions     = captions.to(device)

            logits  = model.forward_cached(raw_features, captions)
            targets = captions[:, 1:]

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            total_loss += loss.item() * raw_features.size(0)
            n          += raw_features.size(0)

    return total_loss / n


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS — UNFROZEN MODE (full forward pass, slower but accurate)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch_full(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n          = 0

    for images, captions in loader:
        images   = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

       
        logits  = model(images, captions)
        targets = captions[:, 1:]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n          += images.size(0)

    return total_loss / n


def evaluate_loss_full(model, loader, criterion, device):
    
    model.eval()
    total_loss = 0.0
    n          = 0

    with torch.no_grad():
        for images, captions in loader:
            images   = images.to(device)
            captions = captions.to(device)

            logits  = model(images, captions)
            targets = captions[:, 1:]

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            total_loss += loss.item() * images.size(0)
            n          += images.size(0)

    return total_loss / n


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(model, loader, vocab, max_len, device):
    
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

            preds = greedy_decode(model, images, max_len, device)

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

    exact_acc = exact_match   / total       if total       > 0 else 0
    token_acc = token_correct / token_total if token_total > 0 else 0
    return exact_acc, token_acc, examples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CNN + Transformer Image Captioning (with backbone fine-tuning)"
    )

    parser.add_argument("--data_dir",   type=str,   default="./dataset")
    parser.add_argument("--embed_dim",  type=int,   default=256)
    parser.add_argument("--num_heads",  type=int,   default=8,
                        help="Must divide embed_dim evenly (e.g. 256/8=32 per head)")
    parser.add_argument("--num_layers", type=int,   default=3)
    parser.add_argument("--ff_dim",     type=int,   default=512)
    parser.add_argument("--max_len",    type=int,   default=25)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Decoder learning rate. Backbone gets lr/10 when unfrozen.")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--save_dir",   type=str,   default="./checkpoints_transformer_final")

    # ── NEW FLAG ──────────────────────────────────────────────────────────────
    parser.add_argument("--unfreeze_backbone", action="store_true", default=False,
                        help=(
                            "Fine-tune the ResNet18 backbone instead of keeping it frozen.\n"
                            "Slower (~45s/epoch vs ~10s/epoch) but much more accurate.\n"
                            "Recommended: train frozen first, then run again with this flag."
                        ))

    args = parser.parse_args()

   
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = (device.type == "cuda")

    print(f"Device: {device}")
    if device.type == "cpu":
        print("  Running on CPU. Google Colab (free T4 GPU) is ~15x faster.")

   
    if args.unfreeze_backbone:
        print("  Mode: UNFROZEN backbone (fine-tuning) — slower but more accurate")
        print("        Backbone LR: {:.2e}  |  Decoder LR: {:.2e}".format(
            args.lr * 0.1, args.lr))
    else:
        print("  Mode: FROZEN backbone (cached) — fast training")

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

    # ── Step 2: Datasets ─────────────────────────────────────────────────────
    print("\n[Step 2] Preparing datasets...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ShapeCaptionDataset(
        data_dir / "train.json", data_dir, vocab, args.max_len, transform)
    val_dataset   = ShapeCaptionDataset(
        data_dir / "val.json",   data_dir, vocab, args.max_len, transform)
    test_dataset  = ShapeCaptionDataset(
        data_dir / "test.json",  data_dir, vocab, args.max_len, transform)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Step 3: Build model ──────────────────────────────────────────────────
    print(f"\n[Step 3] Building CNN + Transformer model...")
    print(f"  embed_dim={args.embed_dim}, num_heads={args.num_heads}, "
          f"num_layers={args.num_layers}, ff_dim={args.ff_dim}")

    # ── CHANGE: pass freeze_backbone = NOT unfreeze_backbone ─────────────────
    model = CNNTransformerCaptioner(
        vocab_size      = vocab_size,
        embed_dim       = args.embed_dim,
        num_heads       = args.num_heads,
        num_layers      = args.num_layers,
        ff_dim          = args.ff_dim,
        max_len         = args.max_len,
        dropout         = args.dropout,
        pretrained      = True,
        freeze_backbone = not args.unfreeze_backbone,  
    ).to(device)


    print("  torch.compile() disabled (not supported on Windows CPU without Visual C++).")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Step 4: Setup loaders and caching ────────────────────────────────────

    if not args.unfreeze_backbone:
        # ── FROZEN MODE: cache features once, train fast ──────────────────────
        print("\n[Step 4] Pre-caching image features (backbone is frozen)...")
        print("  Backbone runs ONCE. Training will skip it every epoch.")

        try:
            backbone = model._orig_mod.encoder.backbone
        except AttributeError:
            backbone = model.encoder.backbone
        backbone = backbone.to(device)

        t_cache      = time.time()
        train_cached = precompute_features(train_dataset, backbone, device, args.batch_size)
        val_cached   = precompute_features(val_dataset,   backbone, device, args.batch_size)
        print(f"  Pre-caching done in {time.time() - t_cache:.1f}s")

        # Cached loaders — yield (raw_features, captions), no images
        train_loader = DataLoader(train_cached, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=0, pin_memory=use_pin_memory)
        val_loader   = DataLoader(val_cached,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    else:
        # ── UNFROZEN MODE: no caching, full forward pass every step ───────────
        print("\n[Step 4] Backbone is UNFROZEN — caching disabled.")
        print("  Backbone runs every training step (gradients flow through it).")
        print("  This is slower but allows backbone to adapt to flat-colour images.")

        # Standard image loaders — yield (images, captions)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=0, pin_memory=use_pin_memory)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    # ── Step 5: Optimiser ────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if args.unfreeze_backbone:
       
        try:
            encoder_params = list(model._orig_mod.encoder.backbone.parameters())
            decoder_params = (list(model._orig_mod.encoder.projection.parameters()) +
                              list(model._orig_mod.encoder.norm.parameters()) +
                              list(model._orig_mod.decoder.parameters()))
        except AttributeError:
            encoder_params = list(model.encoder.backbone.parameters())
            decoder_params = (list(model.encoder.projection.parameters()) +
                              list(model.encoder.norm.parameters()) +
                              list(model.decoder.parameters()))

        optimizer = torch.optim.Adam([
            {"params": encoder_params, "lr": args.lr * 0.1},  
            {"params": decoder_params, "lr": args.lr},        
        ])
        print(f"\n  Optimiser: backbone LR={args.lr*0.1:.1e}, decoder LR={args.lr:.1e}")

    else:
       
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
        print(f"\n  Optimiser: LR={args.lr:.1e} (backbone frozen, not in optimiser)")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Step 6: Training loop ────────────────────────────────────────────────
    mode_label = "UNFROZEN (fine-tuning)" if args.unfreeze_backbone else "FROZEN (cached)"
    print(f"\n[Step 5] Training for {args.epochs} epochs [{mode_label}]...")
    print("-" * 70)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Choose training function based on mode ────────────────────────────
        if args.unfreeze_backbone:
           
            train_loss = train_one_epoch_full(
                model, train_loader, criterion, optimizer, device)
            val_loss   = evaluate_loss_full(
                model, val_loader, criterion, device)
        else:
           
            train_loss = train_one_epoch_cached(
                model, train_loader, criterion, optimizer, device)
            val_loss   = evaluate_loss_cached(
                model, val_loader, criterion, device)

        elapsed = time.time() - t0
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {lr_now:.6f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_name = ("best_transformer_unfrozen.pth" if args.unfreeze_backbone
                         else "best_transformer_frozen.pth")
            ckpt_path = save_dir / ckpt_name

            try:
                state_dict = model._orig_mod.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save({
                "epoch":             epoch,
                "model_state_dict":  state_dict,
                "val_loss":          val_loss,
                "embed_dim":         args.embed_dim,
                "num_heads":         args.num_heads,
                "num_layers":        args.num_layers,
                "ff_dim":            args.ff_dim,
                "max_len":           args.max_len,
                "vocab_size":        vocab_size,
                "freeze_backbone":   not args.unfreeze_backbone,
            }, ckpt_path)
            print(f"  ✓ Best model saved: {ckpt_path}")

    # ── Step 7: Evaluate on test set ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("[Step 6] Evaluating on test set...")

    ckpt_name = ("best_transformer_unfrozen.pth" if args.unfreeze_backbone
                 else "best_transformer_frozen.pth")
    ckpt_path = save_dir / ckpt_name

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            model._orig_mod.load_state_dict(ckpt["model_state_dict"])
        except AttributeError:
            model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best checkpoint "
              f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    exact_acc, token_acc, examples = compute_accuracy(
        model, test_loader, vocab, args.max_len, device
    )

    print(f"\n  Backbone:             {'UNFROZEN (fine-tuned)' if args.unfreeze_backbone else 'FROZEN (cached)'}")
    print(f"  Exact match accuracy: {exact_acc:.4f}  ({exact_acc * 100:.2f}%)")
    print(f"  Token-level accuracy: {token_acc:.4f}  ({token_acc * 100:.2f}%)")

    print(f"\n  Sample predictions:")
    for gt, pred in examples[:5]:
        print(f"    GT:   {gt}")
        print(f"    Pred: {pred}")
        print()

    results = {
        "model":              "CNN + Transformer",
        "backbone":           "unfrozen (fine-tuned)" if args.unfreeze_backbone else "frozen",
        "embed_dim":          args.embed_dim,
        "num_heads":          args.num_heads,
        "num_layers":         args.num_layers,
        "ff_dim":             args.ff_dim,
        "epochs":             args.epochs,
        "best_val_loss":      best_val_loss,
        "exact_match_acc":    exact_acc,
        "token_level_acc":    token_acc,
    }

    results_name = ("results_unfrozen.json" if args.unfreeze_backbone
                    else "results_frozen.json")
    results_path = save_dir / results_name
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
