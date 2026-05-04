import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image as PILImage
from scipy.stats import pearsonr, spearmanr

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────

#CSV_PATH   = "/home/wnp23/cv_project/PKU-AIGIQA-4K/annotation.xlsx"
CSV_PATH   = "/home/wnp23/cv_project/data.csv"

IMAGE_DIR  = "/home/wnp23/cv_project/AGIQA-3K"
CHECKPOINT = "checkpoints/best_msqa_model.pth"
IMAGE_SIZE = 384
BATCH_SIZE = 15
EPOCHS     = 200
LR         = 1e-5
PATIENCE   = 15
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RESUME     = False

os.makedirs("checkpoints", exist_ok=True)
print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────

class AGIQADataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        #img_path = os.path.join(self.image_dir, row["Generated_image"])
        img_path = os.path.join(self.image_dir, row["name"])
        image    = PILImage.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        #mos = torch.tensor(row["MOS_q"], dtype=torch.float32)
        mos = torch.tensor(row["mos_quality"], dtype=torch.float32)
        return image, mos


mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

#df = pd.read_excel(CSV_PATH)[["Generated_image", "MOS_q"]].dropna()
df = pd.read_csv(CSV_PATH)[["name", "mos_quality"]].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n       = len(df)
n_test  = int(n * 0.15)
n_val   = int(n * 0.15)
n_train = n - n_val - n_test

train_df = df.iloc[:n_train]
val_df   = df.iloc[n_train : n_train + n_val]
test_df  = df.iloc[n_train + n_val :]

train_loader = DataLoader(AGIQADataset(train_df, IMAGE_DIR, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(AGIQADataset(val_df, IMAGE_DIR, val_tf),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(AGIQADataset(test_df, IMAGE_DIR, val_tf),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Split -- Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# ── Model ─────────────────────────────────────────────────────────────────────

class MultiScalePooling(nn.Module):
    """Reshape patch tokens into a spatial grid and pool at multiple scales."""

    GRID = IMAGE_SIZE // 16
    SCALES = [1, 2, 6, GRID]

    def __init__(self, dim=1024):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.SCALES])

    def forward(self, patch_tokens):
        B, N, D = patch_tokens.shape
        grid = patch_tokens.transpose(1, 2).reshape(B, D, self.GRID, self.GRID)

        ms_tokens = []
        for scale, norm in zip(self.SCALES, self.norms):
            pooled = F.adaptive_avg_pool2d(grid, scale)
            tokens = pooled.flatten(2).transpose(1, 2)
            ms_tokens.append(norm(tokens))

        return torch.cat(ms_tokens, dim=1)


class QualityAttentionPool(nn.Module):
    """Learnable quality query attends to multi-scale tokens via cross-attention."""

    def __init__(self, dim=1024, num_heads=16, num_queries=1):
        super().__init__()
        self.quality_queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm       = nn.LayerNorm(dim)
        self.ffn        = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, ms_tokens):
        B = ms_tokens.size(0)
        q = self.quality_queries.expand(B, -1, -1)
        attn_out, attn_weights = self.cross_attn(q, ms_tokens, ms_tokens)
        x = self.norm(attn_out + q)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(1), attn_weights


class ViTMSQA(nn.Module):
    """ImageNet ViT-L/16 with Multi-Scale Quality Attention pooling for IQA.

    Positional embeddings are interpolated from 14x14 to the target grid
    when IMAGE_SIZE != 224. Extracts all patch tokens and pools them at
    multiple spatial scales via cross-attention for quality regression.
    """

    def __init__(self, dropout=0.5):
        super().__init__()
        vit = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)

        self.conv_proj   = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder     = vit.encoder
        self.ln          = vit.encoder.ln

        new_grid = IMAGE_SIZE // 16
        old_grid = 14
        if new_grid != old_grid:
            pos_emb = self.encoder.pos_embedding
            cls_pos = pos_emb[:, :1, :]
            patch_pos = pos_emb[:, 1:, :]
            patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid),
                                      mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
            self.encoder.pos_embedding = nn.Parameter(torch.cat([cls_pos, patch_pos], dim=1))

        del vit

        self.ms_pool     = MultiScalePooling(dim=1024)
        self.qa_pool     = QualityAttentionPool(dim=1024, num_heads=16)

        self.head = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def extract_tokens(self, x):
        """Run ViT-L/16 encoder and return (cls_token, patch_tokens)."""
        G = IMAGE_SIZE // 16
        x = self.conv_proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        B = x.shape[0]

        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.encoder(x)

        cls_token = self.ln(x[:, 0])
        patch_tokens = x[:, 1:]
        return cls_token, patch_tokens

    def forward(self, x):
        cls_token, patch_tokens = self.extract_tokens(x)
        ms_tokens = self.ms_pool(patch_tokens)
        qa_repr, self._attn_w = self.qa_pool(ms_tokens)
        fused = torch.cat([cls_token, qa_repr], dim=-1)
        return self.head(fused).squeeze(-1)


model     = ViTMSQA(dropout=0.5).to(DEVICE)
criterion = nn.SmoothL1Loss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

print(f"LR schedule: CosineAnnealingLR  |  LR: {LR:.1e}")

start_epoch = 1
best_val_srcc_resume = -1.0

if RESUME and os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_srcc_resume = ckpt.get("val_srcc", -1.0)
    print(f"Resumed from epoch {ckpt['epoch']} (best SRCC={best_val_srcc_resume:.4f})")
else:
    print("Training from scratch")

total   = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,}  Trainable: {train_p:,}")

ms_token_count = sum(s*s for s in MultiScalePooling.SCALES)
print(f"Multi-scale tokens per image: {ms_token_count} "
      f"(scales {MultiScalePooling.SCALES} -> {[s*s for s in MultiScalePooling.SCALES]})")

# ── Training ──────────────────────────────────────────────────────────────────

history = {"train_loss": [], "val_loss": [], "val_srcc": [], "val_plcc": []}
best_val_srcc = best_val_srcc_resume
epochs_no_improve = 0

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    running_loss, num_samples = 0.0, 0
    for images, scores in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        images, scores = images.to(DEVICE), scores.to(DEVICE)
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, scores)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        num_samples  += images.size(0)

    model.eval()
    val_running, val_n = 0.0, 0
    all_preds, all_gt = [], []
    with torch.no_grad():
        for images, scores in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]  "):
            images, scores = images.to(DEVICE), scores.to(DEVICE)
            preds = model(images)
            val_running += criterion(preds, scores).item() * images.size(0)
            val_n       += images.size(0)
            all_preds.append(preds.cpu().numpy())
            all_gt.append(scores.cpu().numpy())

    train_loss = running_loss / num_samples
    val_loss   = val_running / val_n
    p_arr = np.concatenate(all_preds)
    g_arr = np.concatenate(all_gt)
    val_srcc, _ = spearmanr(p_arr, g_arr)
    val_plcc, _ = pearsonr(p_arr, g_arr)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_srcc"].append(val_srcc)
    history["val_plcc"].append(val_plcc)

    cur_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | SRCC: {val_srcc:.4f} | "
          f"PLCC: {val_plcc:.4f} | LR: {cur_lr:.2e}")

    if val_srcc > best_val_srcc:
        best_val_srcc = val_srcc
        epochs_no_improve = 0
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_srcc": val_srcc}, CHECKPOINT)
        print(f"  -> Saved best model (SRCC={val_srcc:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"Training complete! Best SRCC: {best_val_srcc:.4f}")

# ── Training curves ───────────────────────────────────────────────────────────

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"],   label="Val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
ax1.legend(); ax1.grid(True)

ax2.plot(history["val_srcc"], color="green")
ax2.set_title("Val SRCC"); ax2.set_xlabel("Epoch"); ax2.grid(True)

ax3.plot(history["val_plcc"], color="orange")
ax3.set_title("Val PLCC"); ax3.set_xlabel("Epoch"); ax3.grid(True)

plt.tight_layout()
plt.savefig("training_curves_vit_msqa.png")
plt.show()

# ── Test evaluation ───────────────────────────────────────────────────────────

ckpt = torch.load(CHECKPOINT, weights_only=False)
model.load_state_dict(ckpt["model_state"])
print(f"Loaded best model from epoch {ckpt['epoch']} (SRCC={ckpt['val_srcc']:.4f})\n")

model.eval()
all_preds, all_gt = [], []
with torch.no_grad():
    for images, scores in tqdm(test_loader, desc="Testing"):
        preds = model(images.to(DEVICE)).cpu().numpy()
        all_preds.append(preds)
        all_gt.append(scores.numpy())

preds_arr = np.concatenate(all_preds)
gt_arr    = np.concatenate(all_gt)

srcc, _ = spearmanr(gt_arr, preds_arr)
plcc, _ = pearsonr(gt_arr, preds_arr)
rmse    = np.sqrt(np.mean((gt_arr - preds_arr) ** 2))

print("=" * 40)
print("Test Results (ViT-MSQA)")
print("=" * 40)
print(f"  SRCC : {srcc:.4f}")
print(f"  PLCC : {plcc:.4f}")
print(f"  RMSE : {rmse:.4f}")
print("=" * 40)

plt.figure(figsize=(7, 6))
plt.scatter(gt_arr, preds_arr, alpha=0.4, s=10)
lo, hi = min(gt_arr.min(), preds_arr.min()), max(gt_arr.max(), preds_arr.max())
plt.plot([lo, hi], [lo, hi], "r--")
plt.xlabel("Ground Truth MOS")
plt.ylabel("Predicted MOS")
plt.title(f"ViT-MSQA Test: SRCC={srcc:.4f}  PLCC={plcc:.4f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_scatter_vit_msqa.png")
plt.show()

# ── Attention visualization ───────────────────────────────────────────────────

"""Visualise quality attention weights on a sample image.
Shows which spatial regions (at each scale) the model attends to most."""

model.eval()
sample_img, sample_mos = next(iter(test_loader))
sample_img = sample_img[:1].to(DEVICE)

with torch.no_grad():
    pred = model(sample_img)
    attn_w = model._attn_w[0, 0].cpu().numpy()

scales = MultiScalePooling.SCALES
scale_sizes = [s * s for s in scales]
scale_names = [f"{s}x{s}" for s in scales]

fig, axes = plt.subplots(1, len(scales) + 1, figsize=(4 * (len(scales) + 1), 4))

inv_norm = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)
vis_img = inv_norm(sample_img[0].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
axes[0].imshow(vis_img)
axes[0].set_title(f"GT={sample_mos[0]:.2f}  Pred={pred[0]:.2f}")
axes[0].axis("off")

offset = 0
for i, (s, n, name) in enumerate(zip(scales, scale_sizes, scale_names)):
    w = attn_w[offset : offset + n].reshape(s, s)
    offset += n
    if s == 1:
        axes[i + 1].bar(["global"], [w.item()])
        axes[i + 1].set_ylim(0, attn_w.max() * 1.1)
    else:
        im = axes[i + 1].imshow(w, cmap="hot", interpolation="nearest")
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046)
    axes[i + 1].set_title(f"Scale {name}")

plt.suptitle("Quality Attention Weights by Scale", fontsize=14)
plt.tight_layout()
plt.savefig("attn_viz_msqa.png")
plt.show()
