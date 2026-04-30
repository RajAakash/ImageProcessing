"""
=============================================================================
PKU-AIGIQA-4K  ·  Vision Transformer IQA  ·  Full Pipeline
=============================================================================
USAGE
-----
  # 1. Install dependencies (once)
  pip install torch torchvision scipy scikit-learn pandas matplotlib pillow
             requests tqdm gdown

  # 2. Run the full pipeline
  python pku_aigiqa_vit_pipeline.py

  # 3. (Optional) Skip download if you already have the data
  python pku_aigiqa_vit_pipeline.py --skip-download \
         --img-dir /path/to/images --csv /path/to/annotations.csv

WHAT IT DOES
------------
  Step 1 – Downloads PKU-AIGIQA-4K from the official source
  Step 2 – Parses annotations into a clean CSV (image_name, mos)
  Step 3 – Trains a ViT-B/16 regression model on 70 % of the data
  Step 4 – Validates each epoch; saves the best checkpoint
  Step 5 – Evaluates on the held-out 15 % test set
  Step 6 – Prints MSE / PLCC / SRCC and saves a scatter-plot PNG
=============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────
import os
import sys
import argparse
import zipfile
import shutil
import csv
import json
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party (checked at runtime) ─────────────────────────────────────
def _require(pkg, install_name=None):
    import importlib
    if importlib.util.find_spec(pkg) is None:
        name = install_name or pkg
        print(f"[setup] Installing {name} …")
        os.system(f"{sys.executable} -m pip install -q {name}")

for _p in [("torch",None),("torchvision",None),("scipy",None),
           ("sklearn","scikit-learn"),("pandas",None),
           ("matplotlib",None),("PIL","pillow"),
           ("requests",None),("tqdm",None),("gdown",None)]:
    _require(*_p)

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1 – DATASET DOWNLOAD
# =============================================================================

# PKU-AIGIQA-4K is hosted on the PKU group's Google Drive / GitHub release.
# The canonical download page:  https://github.com/jokingbear/PKU-AIGIQA-4K
#
# Because direct programmatic download requires authentication on some mirrors,
# the script tries three strategies in order:
#   A) gdown (Google Drive direct link)  – works if the folder is public
#   B) wget fallback to a Zenodo/OSF mirror (placeholder URL)
#   C) Manual-mode: the script prints instructions and waits.
#
# Replace GDRIVE_FILE_ID below with the actual Google Drive file ID once you
# confirm the public link from the dataset authors.

DIRECT_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1EuXe_6UNONJSH91uI3edrMMe7utOmpFz&export=download&authuser=0"
)
DATASET_DIR    = Path("pku_aigiqa_4k")
IMG_DIR        = DATASET_DIR / "images"
ANN_PATH       = DATASET_DIR / "annotations.csv"


def _gdrive_download(url: str, dest: Path) -> bool:
    """
    Robustly download a Google Drive file, handling the virus-scan
    confirmation page that Google shows for large files.

    Google Drive may redirect through an HTML warning page before
    serving the actual bytes.  This function detects that page,
    extracts the confirm token (or uuid), and retries automatically.
    """
    import re
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    try:
        print(f"[download] Connecting to Google Drive …")
        resp = session.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            html = resp.text
            # Strategy 1: classic confirm=<token>
            m = re.search(r'confirm=([0-9A-Za-z_\-]+)', html)
            if m:
                token = m.group(1)
                print("[download] Confirm token found; retrying …")
                resp = session.get(url + f"&confirm={token}",
                                   stream=True, timeout=120)
            else:
                # Strategy 2: uuid-based confirmation (newer GDrive)
                m2 = re.search(r'name="uuid"\s+value="([^"]+)"', html)
                if m2:
                    print("[download] UUID confirm found; retrying …")
                    resp = session.get(url + f"&confirm=t&uuid={m2.group(1)}",
                                       stream=True, timeout=120)
                else:
                    # Strategy 3: blind confirm=t
                    print("[download] No token found; trying confirm=t …")
                    resp = session.get(url + "&confirm=t",
                                       stream=True, timeout=120)
            resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        size_str = f"{total/1e6:.1f} MB" if total else "unknown size"
        print(f"[download] Saving to {dest}  ({size_str}) …")

        with open(dest, "wb") as f, tqdm(
                total=total if total else None,
                unit="B", unit_scale=True,
                unit_divisor=1024, desc="dataset.zip") as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))

        # Sanity-check: real ZIP must be > 10 MB
        size = dest.stat().st_size
        if size < 10 * 1024 * 1024:
            print(f"[download] WARNING: file is only {size/1e6:.2f} MB – "
                  "likely an error page, not a real ZIP.")
            dest.unlink()
            return False

        print(f"[download] Complete  ({size/1e6:.1f} MB).")
        return True

    except Exception as e:
        print(f"[download] Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          MANUAL DOWNLOAD REQUIRED  (automated step failed)      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Visit the dataset page:                                      ║
║     https://github.com/jokingbear/PKU-AIGIQA-4K                 ║
║     (or the official download link provided by the authors)      ║
║                                                                  ║
║  2. Download the ZIP archive and place it at:                    ║
║     ./pku_aigiqa_4k/dataset.zip                                  ║
║                                                                  ║
║  3. Re-run this script – it will skip the download step and      ║
║     proceed directly to extraction and training.                 ║
║                                                                  ║
║  Expected ZIP structure (adjust parse_annotations if different): ║
║     dataset.zip                                                  ║
║     ├── images/          (all .jpg / .png files)                 ║
║     └── mos_scores.txt   (or .csv / .json)                       ║
╚══════════════════════════════════════════════════════════════════╝
""")


def maybe_download_dataset(skip: bool = False):
    """Main download orchestrator using the direct Google Drive URL."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / "dataset.zip"

    if skip:
        print("[download] --skip-download flag set; skipping.")
        return

    if zip_path.exists():
        print(f"[download] Archive already present at {zip_path}; skipping download.")
    else:
        print(f"[download] Downloading PKU-AIGIQA-4K …")
        success = _gdrive_download(DIRECT_DOWNLOAD_URL, zip_path)
        if not success:
            print_manual_instructions()
            while not zip_path.exists():
                input("  Press ENTER once you have placed the ZIP at "
                      f"{zip_path} …")
            print("[download] ZIP found – continuing.")

    # ── Extract ──────────────────────────────────────────────────────────
    if not IMG_DIR.exists():
        print(f"[extract] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATASET_DIR)
        # The ZIP unpacks into a single sub-folder (e.g. PKU-AIGIQA-4K/).
        # Before renaming it to "images/", rescue any annotation files
        # that live at the top level of that sub-folder.
        subdirs = [d for d in DATASET_DIR.iterdir()
                   if d.is_dir() and d.name != "images"]
        if subdirs and not IMG_DIR.exists():
            extracted_root = subdirs[0]
            # Move annotation spreadsheets up so they survive the rename
            for ann in (list(extracted_root.glob("*.xlsx")) +
                        list(extracted_root.glob("*.xls"))  +
                        list(extracted_root.glob("*.csv"))  +
                        list(extracted_root.glob("*.txt"))):
                dest = DATASET_DIR / ann.name
                if not dest.exists():
                    shutil.move(str(ann), str(dest))
                    print(f"[extract] Rescued annotation file -> {dest}")
            print(f"[extract] Renaming {extracted_root} -> {IMG_DIR}")
            extracted_root.rename(IMG_DIR)
        print("[extract] Done.")
    else:
        print("[extract] Images directory already exists; skipping extraction.")



# =============================================================================
# SECTION 2 – ANNOTATION PARSING
# =============================================================================

def _parse_xlsx(path):
    """
    Parse the PKU-AIGIQA-4K annotation.xlsx file.
    Robustly finds image-name and MOS columns regardless of exact header names.
    """
    try:
        import openpyxl  # noqa
    except ImportError:
        print("[annotations] Installing openpyxl for .xlsx support …")
        os.system(f"{sys.executable} -m pip install -q openpyxl")

    print(f"[annotations] Reading {path} …")
    df = pd.read_excel(path, engine="openpyxl")

    # Normalise column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    print(f"[annotations] Columns found: {list(df.columns)}")

    # Find image-name column
    # PKU-AIGIQA-4K uses 'generated_image' as the image filename column
    img_col = None
    for c in ["generated_image","image_name","imagename","image","filename",
              "file_name","name","img_name","img"]:
        if c in df.columns:
            img_col = c; break
    if img_col is None:
        for col in df.columns:
            if df[col].dtype == object:
                img_col = col; break
    if img_col is None:
        raise ValueError(f"Cannot find image-name column. Columns: {list(df.columns)}")

    # Find MOS column
    # PKU-AIGIQA-4K has three MOS scores:
    #   mos_q = perceptual quality (how good the image looks)
    #   mos_a = aesthetic score
    #   mos_c = text-image correspondence
    # We use mos_q (quality) as our regression target per the paper.
    mos_col = None
    for c in ["mos_q","mos","mos_score","score","quality_score",
              "mean_opinion_score","rating","quality"]:
        if c in df.columns:
            mos_col = c; break
    if mos_col is None:
        for col in df.columns:
            if col != img_col and pd.api.types.is_numeric_dtype(df[col]):
                mos_col = col; break
    if mos_col is None:
        raise ValueError(f"Cannot find MOS column. Columns: {list(df.columns)}")

    print(f"[annotations] Using  image_col=\'{img_col}\'  mos_col=\'{mos_col}\'")

    result = pd.DataFrame({
        "image_name": df[img_col].astype(str).str.strip(),
        "mos":        pd.to_numeric(df[mos_col], errors="coerce"),
    }).dropna(subset=["mos"]).reset_index(drop=True)

    # Auto-fix missing extensions using rglob (images may be in subdirs)
    img_dir_files = ({f.name for f in IMG_DIR.rglob("*")
                      if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}}
                     if IMG_DIR.exists() else set())
    def _fix_ext(name):
        p = Path(name)
        if p.name in img_dir_files:
            return name
        for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
            if (p.stem + ext) in img_dir_files:
                return p.stem + ext
        return name
    result["image_name"] = result["image_name"].apply(_fix_ext)

    print(f"[annotations] Parsed {len(result)} entries from xlsx.")
    return result


def parse_annotations(ann_file=None):
    """
    Build a DataFrame with columns [image_name, mos].
    Priority: explicit path → .xlsx in DATASET_DIR → csv/json/txt fallbacks.
    """
    ann_file = Path(ann_file) if ann_file else None

    # 1. Explicit path
    if ann_file and ann_file.exists():
        if ann_file.suffix.lower() in {".xlsx", ".xls"}:
            return _parse_xlsx(ann_file)
        df = pd.read_csv(ann_file)
        df.columns = [c.strip().lower() for c in df.columns]
        if "image_name" not in df.columns or "mos" not in df.columns:
            raise ValueError(f"CSV needs image_name and mos columns. Found: {list(df.columns)}")
        return df

    # 2. Search for .xlsx inside DATASET_DIR (PKU-AIGIQA-4K ships annotation.xlsx)
    # rglob finds all .xlsx including temp lock files (~$filename.xlsx) — skip those
    xlsx_files = [p for p in
                  list(DATASET_DIR.rglob("*.xlsx")) + list(DATASET_DIR.rglob("*.xls"))
                  if not p.name.startswith("~$")]
    if xlsx_files:
        def _priority(p):
            kws = ["annot","mos","score","label","rating","quality"]
            return 0 if any(k in p.stem.lower() for k in kws) else 1
        xlsx_files.sort(key=_priority)
        chosen = xlsx_files[0]
        print(f"[annotations] Found xlsx: {chosen}")
        return _parse_xlsx(chosen)

    # 3. CSV / JSON / TXT fallbacks
    candidates = (list(DATASET_DIR.rglob("*.csv")) +
                  list(DATASET_DIR.rglob("*.json")) +
                  list(DATASET_DIR.rglob("*.txt")))
    records = []
    for candidate in candidates:
        stem = candidate.stem.lower()
        if not any(kw in stem for kw in ["mos","score","label","annot","rating"]):
            continue
        print(f"[annotations] Parsing {candidate}")
        if candidate.suffix == ".csv":
            df = pd.read_csv(candidate)
            df.columns = [c.strip().lower() for c in df.columns]
            return df
        elif candidate.suffix == ".json":
            with open(candidate) as f:
                data = json.load(f)
            if isinstance(data, list):
                records = [{"image_name": d.get("image_name", d.get("name")),
                            "mos": float(d.get("mos", d.get("score")))} for d in data]
            else:
                records = [{"image_name": k, "mos": float(v)} for k, v in data.items()]
            return pd.DataFrame(records)
        else:
            with open(candidate) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            records.append({"image_name": parts[0], "mos": float(parts[-1])})
                        except ValueError:
                            continue
            if records:
                return pd.DataFrame(records)

    raise FileNotFoundError(
        f"No annotation file found in {DATASET_DIR}.\n"
        "Pass it explicitly:  --xlsx path/to/annotation.xlsx")



def build_csv(df: pd.DataFrame, out_path: Path = ANN_PATH) -> pd.DataFrame:
    """
    Resolve image paths and drop rows whose image cannot be found on disk.

    Handles three cases:
      1. annotation stores bare filename   : 'img_001.jpg'
      2. annotation stores relative path   : 'subdir/img_001.jpg'
      3. annotation stores name without ext: 'img_001'
    """
    if not IMG_DIR.exists():
        df.to_csv(out_path, index=False)
        return df

    # Build two lookup maps from everything on disk
    # name_to_rel : basename (lower) -> path relative to IMG_DIR
    # stem_to_rel : stem (lower)     -> path relative to IMG_DIR
    name_to_rel = {}
    stem_to_rel = {}
    for f in IMG_DIR.rglob("*"):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            rel = str(f.relative_to(IMG_DIR))
            name_to_rel[f.name.lower()] = rel
            stem_to_rel[f.stem.lower()] = rel

    print(f"[annotations] Indexed {len(name_to_rel)} images on disk.")

    def _resolve(raw: str) -> str | None:
        raw = raw.strip()
        p = Path(raw)

        # Case 1: exact relative path exists
        if (IMG_DIR / raw).exists():
            return raw

        # Case 2: just the filename matches
        key = p.name.lower()
        if key in name_to_rel:
            return name_to_rel[key]

        # Case 3: stem match (no extension in annotation)
        key_stem = p.stem.lower()
        if key_stem in stem_to_rel:
            return stem_to_rel[key_stem]

        return None  # not found

    before = len(df)
    df = df.copy()
    df["image_name"] = df["image_name"].apply(_resolve)
    df = df[df["image_name"].notna()].reset_index(drop=True)
    print(f"[annotations] Matched {len(df)}/{before} images on disk "
          f"(dropped {before - len(df)} unresolved).")

    if len(df) == 0:
        # Print a sample to help debug
        sample = open(str(out_path).replace("annotations.csv","_debug_sample.txt"), "w")
        orig = pd.read_excel(str(out_path).replace("annotations.csv","images/annotation.xlsx"),
                             engine="openpyxl") if False else None
        print("[annotations] ERROR: 0 images matched. "
              "Check that IMG_DIR points to the right folder.")
        disk_samples = list(name_to_rel.values())[:5]
        print(f"  Disk samples : {disk_samples}")

    df.to_csv(out_path, index=False)
    print(f"[annotations] Saved to {out_path}")
    return df


# =============================================================================
# SECTION 3 – PYTORCH DATASET
# =============================================================================

class PKUAIGIQADataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        # image_name may be a relative subpath like "subdir/img.jpg"
        img_path = self.img_dir / self.df.loc[idx, "image_name"]
        try:
            image = PILImage.open(img_path).convert("RGB")
        except Exception as e:
            # Return a black image if file is corrupted / missing
            image = PILImage.new("RGB", (224, 224), (0, 0, 0))
        mos = torch.tensor(float(self.df.loc[idx, "mos"]), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, mos


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(df, img_dir, batch_size=32, num_workers=4, seed=42):
    # Normalise MOS to [0, 1]
    mos_min = df["mos"].min()
    mos_max = df["mos"].max()
    df      = df.copy()
    df["mos"] = (df["mos"] - mos_min) / (mos_max - mos_min)

    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=seed)
    val_df,  test_df = train_test_split(tmp_df, test_size=0.50, random_state=seed)

    print(f"[data] Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    def _loader(split_df, train):
        ds = PKUAIGIQADataset(split_df, img_dir,
                              transform=get_transforms(train=train))
        return DataLoader(ds, batch_size=batch_size, shuffle=train,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=train)

    return (_loader(train_df, True),
            _loader(val_df,   False),
            _loader(test_df,  False),
            (mos_min, mos_max))


# =============================================================================
# SECTION 4 – MODEL
# =============================================================================

class ViTIQA(nn.Module):
    """
    ViT-B/16 backbone with a lightweight regression head.

    The CLS token representation (dim=768) is passed through:
        Linear(768→256) → GELU → Dropout → Linear(256→64)
        → GELU → Dropout → Linear(64→1) → Sigmoid
    Sigmoid keeps outputs in [0,1] matching the normalised MOS targets.
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        embed_dim = backbone.heads.head.in_features     # 768
        backbone.heads = nn.Identity()
        self.backbone = backbone

         # 👉 ADD IT HERE (freeze most layers)
        for name, param in self.backbone.named_parameters():
            if not any(layer in name for layer in [
                "encoder.layers.6", "encoder.layers.7",
                "encoder.layers.8", "encoder.layers.9",
                "encoder.layers.10", "encoder.layers.11"]):
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)   # (B,)


# =============================================================================
# SECTION 5 – TRAINING UTILITIES
# =============================================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mse          = float(np.mean((preds - targets) ** 2))
    plcc, _      = pearsonr(preds, targets)
    srcc, _      = spearmanr(preds, targets)
    return {"MSE": round(mse, 6), "PLCC": round(float(plcc), 4),
            "SRCC": round(float(srcc), 4)}


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = -math.inf

    def step(self, value: float) -> bool:
        if value > self.best + self.min_delta:
            self.best    = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []

    with torch.set_grad_enabled(training):
        for imgs, mos in loader:
            imgs, mos = imgs.to(device), mos.to(device)
            out  = model(imgs)
            loss = criterion(out, mos)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss     += loss.item() * imgs.size(0)
            preds_all.extend(out.detach().cpu().numpy())
            targets_all.extend(mos.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(np.array(preds_all), np.array(targets_all))
    return avg_loss, metrics


# =============================================================================
# SECTION 6 – FULL TRAINING LOOP
# =============================================================================

def train_model(model, train_loader, val_loader, device, cfg: dict):
    criterion  = nn.MSELoss()
    optimizer  = optim.AdamW(model.parameters(),
                             lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=cfg["epochs"])
    stopper    = EarlyStopping(patience=cfg["patience"])

    best_srcc    = -1.0
    best_weights = None
    history      = {"train_loss": [], "val_loss": [],
                    "val_plcc":   [], "val_srcc": []}

    header = (f"{'Ep':>4} | {'TrainLoss':>9} | {'ValLoss':>8} | "
              f"{'PLCC':>6} | {'SRCC':>6} | {'LR':>8}")
    print("\n" + header)
    print("─" * len(header))

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, _   = run_epoch(model, train_loader, criterion,
                                 optimizer, device, training=True)
        val_loss, vm = run_epoch(model, val_loader, criterion,
                                 optimizer, device, training=False)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_plcc"].append(vm["PLCC"])
        history["val_srcc"].append(vm["SRCC"])

        flag = ""
        if vm["SRCC"] > best_srcc:
            best_srcc    = vm["SRCC"]
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, cfg["ckpt"])
            flag = "  ✓ saved"

        elapsed = time.time() - t0
        print(f"{epoch:>4} | {tr_loss:>9.5f} | {val_loss:>8.5f} | "
              f"{vm['PLCC']:>6.4f} | {vm['SRCC']:>6.4f} | "
              f"{lr:>8.2e}  ({elapsed:.1f}s){flag}")

        if stopper.step(vm["SRCC"]):
            print(f"\n[train] Early stopping at epoch {epoch} "
                  f"(best SRCC={best_srcc:.4f}).")
            break

    model.load_state_dict(best_weights)
    return model, history


# =============================================================================
# SECTION 7 – EVALUATION & VISUALISATION
# =============================================================================

def evaluate_model(model, test_loader, device,
                   mos_range, out_dir: Path):
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for imgs, mos in test_loader:
            out = model(imgs.to(device)).cpu().numpy()
            preds_all.extend(out)
            targets_all.extend(mos.numpy())

    preds   = np.array(preds_all)
    targets = np.array(targets_all)

    # Denormalise back to original MOS scale
    mos_min, mos_max = mos_range
    preds_orig   = preds   * (mos_max - mos_min) + mos_min
    targets_orig = targets * (mos_max - mos_min) + mos_min

    metrics = compute_metrics(preds_orig, targets_orig)

    print("\n" + "═" * 40)
    print("  TEST SET RESULTS  (original MOS scale)")
    print("═" * 40)
    for k, v in metrics.items():
        print(f"  {k:>6}: {v}")
    print("═" * 40 + "\n")

    # ── Scatter plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets_orig, preds_orig, alpha=0.35, s=8, color="#4C72B0")
    lo = min(targets_orig.min(), preds_orig.min())
    hi = max(targets_orig.max(), preds_orig.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Ground-Truth MOS", fontsize=12)
    ax.set_ylabel("Predicted MOS",    fontsize=12)
    ax.set_title(
        f"ViT-B/16  NR-IQA  |  PKU-AIGIQA-4K\n"
        f"PLCC={metrics['PLCC']:.4f}   SRCC={metrics['SRCC']:.4f}   "
        f"MSE={metrics['MSE']:.4f}", fontsize=10)
    ax.legend()
    plt.tight_layout()
    scatter_path = out_dir / "scatter_vit_iq.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"[eval] Scatter plot saved → {scatter_path}")

    return metrics


def plot_training_curves(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend()

    axes[1].plot(epochs, history["val_plcc"], label="Val PLCC")
    axes[1].plot(epochs, history["val_srcc"], label="Val SRCC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Correlation")
    axes[1].set_title("Validation Metrics"); axes[1].legend()

    plt.tight_layout()
    curve_path = out_dir / "training_curves_vit.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[eval] Training curves saved → {curve_path}")


# =============================================================================
# SECTION 8 – MAIN ENTRY POINT
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PKU-AIGIQA-4K  Vision Transformer IQA Pipeline")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download step (use existing data)")
    p.add_argument("--img-dir",  type=str, default=str(IMG_DIR),
                   help="Path to image directory")
    p.add_argument("--csv",      type=str, default=None,
                   help="Path to annotation CSV (image_name, mos)")
    p.add_argument("--xlsx",     type=str, default=None,
                   help="Path to annotation XLSX (auto-detected if omitted)")
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--lr",       type=float, default=3e-5)
    p.add_argument("--dropout",  type=float, default=0.3)
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--ckpt",     type=str, default="checkpoints/vit_iq_best.pth")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Output directories ────────────────────────────────────────────────
    out_dir = Path("results");       out_dir.mkdir(exist_ok=True)
    ckpt_dir = Path(args.ckpt).parent; ckpt_dir.mkdir(exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")
    print(f"[main] Device: {device}")

    # ── Step 1: Download ──────────────────────────────────────────────────
    maybe_download_dataset(skip=args.skip_download)

    # ── Step 2: Annotations ───────────────────────────────────────────────
    ann_file = Path(args.xlsx) if args.xlsx else (Path(args.csv) if args.csv else None)
    df = parse_annotations(ann_file)
    df = build_csv(df)
    print(f"[main] Dataset: {len(df)} images  |  "
          f"MOS range [{df['mos'].min():.2f}, {df['mos'].max():.2f}]")

    # ── Step 3: DataLoaders ───────────────────────────────────────────────
    train_loader, val_loader, test_loader, mos_range = get_dataloaders(
        df,
        img_dir    = args.img_dir,
        batch_size = args.batch,
        num_workers= args.workers,
        seed       = args.seed,
    )

    # ── Step 4: Model ─────────────────────────────────────────────────────
    model = ViTIQA(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Trainable parameters: {n_params:,}")

    cfg = {
        "epochs"      : args.epochs,
        "lr"          : args.lr,
        "weight_decay": 1e-3,
        "patience"    : 10,
        "ckpt"        : args.ckpt,
    }

    # ── Step 5: Train ─────────────────────────────────────────────────────
    model, history = train_model(model, train_loader, val_loader, device, cfg)
    plot_training_curves(history, out_dir)

    # ── Step 6: Evaluate ──────────────────────────────────────────────────
    metrics = evaluate_model(model, test_loader, device, mos_range, out_dir)

    # Save metrics to JSON for easy reference
    metrics_path = out_dir / "test_metrics_vit.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[main] Metrics saved → {metrics_path}")
    print("[main] Pipeline complete. ✓")


if __name__ == "__main__":
    main()