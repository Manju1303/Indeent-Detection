"""
models/train_violence.py
Complete training script for the MobileNetV2+LSTM violence classifier.
Trains on RWF-2000, Hockey Fight, or any custom violence dataset.

Dataset structure expected:
  data/violence_dataset/
    train/
      fight/       ← video files or frame folders
      non-fight/
    val/
      fight/
      non-fight/

Free Datasets:
  - RWF-2000:      https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
  - Hockey Fight:  https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes
  - UCF Crime:     https://webpages.charlotte.edu/cchen62/dataset.html

Usage:
  python models/train_violence.py --data_dir data/violence_dataset --epochs 30
"""

import os
import sys
import argparse
import logging
import time
import json
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Argument parsing ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train violence detection LSTM model")
    p.add_argument("--data_dir",    default="data/violence_dataset", help="Dataset root")
    p.add_argument("--output_dir",  default="models",                help="Where to save weights")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--seq_len",     type=int, default=16,  help="Frames per sequence")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--hidden_dim",  type=int, default=256)
    p.add_argument("--num_layers",  type=int, default=2)
    p.add_argument("--device",      default="auto")
    p.add_argument("--frame_size",  type=int, default=224)
    p.add_argument("--workers",     type=int, default=2)
    return p.parse_args()


# ── Dataset ──────────────────────────────────────────────────

class ViolenceVideoDataset:
    """Loads video clips as frame sequences for training."""

    LABEL_MAP = {"fight": 1, "non-fight": 0, "violence": 1, "normal": 0}

    def __init__(self, root: str, split: str, seq_len: int, frame_size: int):
        try:
            import torch
            import cv2
            from torchvision import transforms
        except ImportError as e:
            raise RuntimeError(f"Missing dependency: {e}")

        import torch
        self.seq_len    = seq_len
        self.frame_size = frame_size
        self.samples    = []   # list of (video_path_or_dir, label)
        self.transform  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            logger.error(f"Dataset split not found: {split_dir}")
            return

        for class_name in os.listdir(split_dir):
            label = self.LABEL_MAP.get(class_name.lower())
            if label is None:
                logger.warning(f"Unknown class folder: {class_name} (skipping)")
                continue
            class_dir = os.path.join(split_dir, class_name)
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if fname.endswith((".mp4", ".avi", ".mov", ".mkv")) or os.path.isdir(fpath):
                    self.samples.append((fpath, label))

        logger.info(f"[Dataset] {split}: {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        import cv2
        from PIL import Image

        path, label = self.samples[idx]
        frames = self._load_frames(path)
        tensors = []
        for frame in frames:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil = pil.resize((self.frame_size, self.frame_size))
            tensors.append(self.transform(pil))
        sequence = torch.stack(tensors)    # (T, C, H, W)
        return sequence, label

    def _load_frames(self, path: str):
        import cv2
        frames = []
        if os.path.isdir(path):
            # Frame folder: sorted images
            imgs = sorted([f for f in os.listdir(path) if f.endswith((".jpg", ".png"))])
            for img in imgs[:self.seq_len * 4]:   # sample more, then subsample
                frame = cv2.imread(os.path.join(path, img))
                if frame is not None:
                    frames.append(frame)
        else:
            # Video file
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                step = max(1, total // (self.seq_len * 2))
                for i in range(0, min(total, self.seq_len * step), step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            cap.release()

        # Subsample to exactly seq_len frames
        if len(frames) == 0:
            # Return blank frames as fallback
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.seq_len
        elif len(frames) < self.seq_len:
            # Pad by repeating last frame
            frames += [frames[-1]] * (self.seq_len - len(frames))
        else:
            # Uniformly subsample
            indices = np.linspace(0, len(frames) - 1, self.seq_len, dtype=int)
            frames  = [frames[i] for i in indices]

        return frames[:self.seq_len]


# ── Model ────────────────────────────────────────────────────

def build_model(feat_dim: int, hidden_dim: int, num_layers: int):
    import torch.nn as nn

    class ViolenceLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0.3)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return ViolenceLSTM()


# ── Training loop ────────────────────────────────────────────

def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import models

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Training on: {device}")

    # Feature extractor (frozen backbone)
    backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
    backbone.classifier = nn.Identity()
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)

    # Get feature dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.frame_size, args.frame_size).to(device)
        feat_dim = backbone(dummy).shape[1]
    logger.info(f"Feature dim: {feat_dim}")

    # LSTM model
    model = build_model(feat_dim, args.hidden_dim, args.num_layers).to(device)

    # Dataset
    train_ds = ViolenceVideoDataset(args.data_dir, "train", args.seq_len, args.frame_size)
    val_ds   = ViolenceVideoDataset(args.data_dir, "val",   args.seq_len, args.frame_size)

    if len(train_ds) == 0:
        logger.error("No training data found. Check your data_dir structure.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    history      = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss, train_correct, n = 0.0, 0, 0

        for sequences, labels in train_loader:
            # sequences: (B, T, C, H, W)
            B, T, C, H, W = sequences.shape
            sequences = sequences.to(device).float()
            labels    = labels.to(device)

            # Extract features frame-by-frame
            seq_flat = sequences.view(B * T, C, H, W)
            with torch.no_grad():
                feats = backbone(seq_flat)       # (B*T, feat_dim)
            feats = feats.view(B, T, -1)         # (B, T, feat_dim)

            optimizer.zero_grad()
            output = model(feats)
            loss   = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item() * B
            train_correct += (output.argmax(1) == labels).sum().item()
            n             += B

        train_acc  = train_correct / n
        train_loss = train_loss / n

        # ── Validate ───────────────────────────────────────
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                B, T, C, H, W = sequences.shape
                sequences = sequences.to(device).float()
                labels    = labels.to(device)
                seq_flat  = sequences.view(B * T, C, H, W)
                feats     = backbone(seq_flat).view(B, T, -1)
                output    = model(feats)
                val_correct += (output.argmax(1) == labels).sum().item()
                val_n       += B

        val_acc = val_correct / val_n if val_n > 0 else 0
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Acc: {val_acc:.2%}"
        )

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc})

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "violence_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Best model saved (val_acc={val_acc:.2%})")

    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n✅ Training complete. Best val accuracy: {best_val_acc:.2%}")
    logger.info(f"   Model saved to: {os.path.join(args.output_dir, 'violence_model.pth')}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
