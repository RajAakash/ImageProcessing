"""
=============================================================================
PKU-AIGIQA-4K  —  Figure Generation Script
=============================================================================
Generates 9 analysis figures matching the style of the AGIQA-3K reference
figures, adapted to the PKU-AIGIQA-4K annotation schema:

    Columns expected in annotation.xlsx / annotations.csv:
        image_prompt     : text prompt used for generation
        text_prompt      : (may overlap with image_prompt)
        generated_image  : filename of the image
        mos_q            : perceptual quality MOS  ← primary target
        mos_a            : alignment / correspondence MOS
        mos_c            : aesthetic MOS

    Figures produced:
        fig1_mos_overview.png       — MOS distributions, boxplot, scatter, stats
        fig2_per_model.png          — per-model bar + violin (model inferred from filename)
        fig3_style_adj.png          — MOS by style + top adjectives from prompts
        fig4_heatmap.png            — model × style MOS heatmap
        fig5_reliability.png        — MOS vs STD scatter + correlation matrix
        fig6_samples.png            — sample images at low / mid / high MOS
        fig7_feature_analysis.png   — image feature ↔ MOS correlations + PCA
        fig8_ml_results.png         — ML cross-validation (Ridge / RF / GBM)
        fig9_feature_importance.png — Random Forest feature importance

USAGE
-----
    python generate_figures.py                              # uses defaults
    python generate_figures.py --xlsx pku_aigiqa_4k/images/annotation.xlsx \
                               --img-dir pku_aigiqa_4k/images \
                               --out-dir figures
"""

# ── stdlib ────────────────────────────────────────────────────────────────
import os
import re
import sys
import warnings
import argparse
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.stats import pearsonr, spearmanr
from PIL import Image as PILImage

# sklearn (for ML figures)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_validate, KFold
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] scikit-learn not found — fig7/fig8/fig9 will be skipped. "
          "Install with: pip install scikit-learn")

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

BLUE   = "#4472C4"
PURPLE = "#6A0DAD"
PINK   = "#E91E8C"
CYAN   = "#00BCD4"
DARK   = "#1A1A2E"

# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_data(xlsx_path: str = None, csv_path: str = None) -> pd.DataFrame:
    """Load and normalise the PKU-AIGIQA-4K annotation file."""
    if xlsx_path and Path(xlsx_path).exists():
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    elif csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        # Auto-detect
        for p in ["pku_aigiqa_4k/annotations.csv",
                  "pku_aigiqa_4k/images/annotation.xlsx",
                  "annotations.csv", "annotation.xlsx"]:
            if Path(p).exists():
                df = pd.read_excel(p, engine="openpyxl") \
                     if p.endswith(".xlsx") else pd.read_csv(p)
                print(f"[data] Loaded {p}")
                break
        else:
            raise FileNotFoundError(
                "No annotation file found. Pass --xlsx or --csv explicitly.")

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map PKU column names → standard internal names
    col_map = {
        "generated_image": "image_name",
        "image_prompt":    "image_prompt",
        "text_prompt":     "text_prompt",
        "mos_q":           "mos_q",
        "mos_a":           "mos_a",
        "mos_c":           "mos_c",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure we have at least mos_q
    if "mos_q" not in df.columns:
        raise ValueError(
            f"Expected 'mos_q' column. Found: {list(df.columns)}")

    # Drop rows with missing MOS
    df = df.dropna(subset=["mos_q"]).reset_index(drop=True)
    print(f"[data] {len(df)} images loaded. "
          f"MOS-Q range: [{df['mos_q'].min():.3f}, {df['mos_q'].max():.3f}]")
    return df


def infer_model(image_name: str) -> str:
    """
    Infer the generating model from the image filename.
    PKU-AIGIQA-4K filenames typically start with model codes like:
        mj_  / midjourney_  → Midjourney
        sd_  / stable_      → Stable Diffusion
        dalle_ / dall-e     → DALL-E 3
        Other               → Unknown
    Adjust the patterns below to match your actual filenames.
    """
    n = str(image_name).lower()
    if any(k in n for k in ["mj_", "midjourney"]):
        return "Midjourney"
    if any(k in n for k in ["sd_", "stable", "sdxl"]):
        return "Stable Diffusion"
    if any(k in n for k in ["dalle", "dall-e", "de3"]):
        return "DALL-E 3"
    # Fallback: use first 2-3 chars of filename as model code
    stem = Path(n).stem
    prefix = re.sub(r"[^a-z]", "", stem)[:4]
    return prefix if prefix else "unknown"


def infer_style(prompt: str) -> str:
    """Extract style keyword from the prompt text."""
    if pd.isna(prompt):
        return "unspecified"
    p = str(prompt).lower()
    for style in ["baroque", "realistic", "sci-fi", "anime", "abstract",
                  "fantasy", "vintage", "minimalist", "surreal", "cartoon"]:
        if style in p:
            return style + " style"
    return "unspecified"


def extract_adjectives(prompts: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """Extract most frequent multi-word descriptors from prompts."""
    phrases = []
    for prompt in prompts.dropna():
        words = str(prompt).lower().split()
        # bigrams
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
    counts = Counter(phrases).most_common(top_n * 3)
    # filter out boring combos
    stopwords = {"the a", "a the", "of the", "in the", "on the", "and the",
                 "with a", "a very", "is a", "it is"}
    result = [(p, c) for p, c in counts if p not in stopwords][:top_n]
    return pd.DataFrame(result, columns=["phrase", "count"])


# =============================================================================
# SECTION 2 — IMAGE FEATURE EXTRACTION
# =============================================================================

def extract_image_features(df: pd.DataFrame, img_dir: str,
                            max_images: int = 1000) -> pd.DataFrame:
    """
    Extract per-image low-level features from the actual image files.
    Returns a DataFrame aligned with df (rows where extraction succeeded).
    """
    img_dir = Path(img_dir)
    records = []
    n = min(len(df), max_images)
    print(f"[features] Extracting features from {n} images …")

    for idx, row in df.iloc[:n].iterrows():
        img_path = img_dir / str(row.get("image_name", ""))
        try:
            img = PILImage.open(img_path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

            # Sharpness via Laplacian variance
            gray = 0.299*r + 0.587*g + 0.114*b
            lap = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
            from scipy.ndimage import convolve
            sharpness = float(np.var(convolve(gray, lap)))

            # Edge density
            gx = np.diff(gray, axis=1)
            gy = np.diff(gray, axis=0)
            edge_density = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))

            # Entropy
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,256))
            hist = hist / hist.sum()
            entropy = float(-np.sum(hist[hist>0] * np.log2(hist[hist>0])))

            records.append({
                "idx":         idx,
                "width":       img.width,
                "height":      img.height,
                "aspect_ratio":img.width / max(img.height, 1),
                "brightness":  float(gray.mean()),
                "r_mean":      float(r.mean()),
                "g_mean":      float(g.mean()),
                "b_mean":      float(b.mean()),
                "r_std":       float(r.std()),
                "g_std":       float(g.std()),
                "b_std":       float(b.std()),
                "contrast":    float(gray.std()),
                "saturation":  float(np.std([r,g,b])),
                "colorfulness":float(np.sqrt(np.var(r-g) + np.var(r+g-2*b)) * 0.45
                                     + np.mean(np.sqrt((r-g)**2 + (r+g-2*b)**2)) * 0.56),
                "sharpness":   sharpness,
                "edge_density":edge_density,
                "entropy":     entropy,
                "mos_q":       row["mos_q"],
                "mos_a":       row.get("mos_a", np.nan),
            })
        except Exception:
            continue

    feat_df = pd.DataFrame(records).set_index("idx")
    print(f"[features] Extracted {len(feat_df)} images successfully.")
    return feat_df


# =============================================================================
# SECTION 3 — FIGURE GENERATION
# =============================================================================

# ── Fig 1: MOS Overview ──────────────────────────────────────────────────────
def fig1_mos_overview(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("PKU-AIGIQA-4K — MOS Score Overview",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    mos_q = df["mos_q"].dropna()
    mos_a = df["mos_a"].dropna() if "mos_a" in df.columns else pd.Series(dtype=float)
    mos_c = df["mos_c"].dropna() if "mos_c" in df.columns else pd.Series(dtype=float)

    # --- Perception (mos_q) histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(mos_q, bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax0.axvline(mos_q.mean(),   color="crimson",  linestyle="--", lw=1.5,
                label=f"μ={mos_q.mean():.3f}")
    ax0.axvline(mos_q.median(), color="darkviolet",linestyle=":",  lw=1.5,
                label=f"med={mos_q.median():.3f}")
    ax0.set_title("Quality MOS Distribution (mos_q)", fontsize=11)
    ax0.set_xlabel("MOS"); ax0.set_ylabel("Count")
    ax0.legend(fontsize=8)

    # --- Alignment (mos_a) histogram
    ax1 = fig.add_subplot(gs[0, 1])
    if len(mos_a):
        ax1.hist(mos_a, bins=40, color=PURPLE, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax1.axvline(mos_a.mean(),   color="crimson",  linestyle="--", lw=1.5,
                    label=f"μ={mos_a.mean():.3f}")
        ax1.axvline(mos_a.median(), color="gold",      linestyle=":",  lw=1.5,
                    label=f"med={mos_a.median():.3f}")
        ax1.legend(fontsize=8)
    ax1.set_title("Alignment MOS Distribution (mos_a)", fontsize=11)
    ax1.set_xlabel("MOS"); ax1.set_ylabel("Count")

    # --- Scatter: mos_q vs mos_a
    ax2 = fig.add_subplot(gs[0, 2])
    if len(mos_a) and len(mos_q):
        shared = df[["mos_q","mos_a"]].dropna()
        r, p = pearsonr(shared["mos_q"], shared["mos_a"])
        ax2.scatter(shared["mos_q"], shared["mos_a"],
                    alpha=0.25, s=6, color=BLUE)
        ax2.set_title(f"Quality vs Alignment\n(Pearson r={r:.3f}, p={p:.2e})",
                      fontsize=10)
        ax2.set_xlabel("MOS Quality"); ax2.set_ylabel("MOS Alignment")

    # --- Boxplot comparison
    ax3 = fig.add_subplot(gs[1, 0])
    box_data = [mos_q]
    box_labels = ["mos_q"]
    colors_box = [BLUE]
    for col, lbl, clr in [("mos_a", "mos_a", PURPLE), ("mos_c", "mos_c", CYAN)]:
        if col in df.columns and df[col].notna().sum() > 0:
            box_data.append(df[col].dropna())
            box_labels.append(lbl)
            colors_box.append(clr)
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True,
                     medianprops={"color":"black","linewidth":2})
    for patch, clr in zip(bp["boxes"], colors_box):
        patch.set_facecolor(clr); patch.set_alpha(0.7)
    ax3.set_title("MOS Boxplot Comparison", fontsize=11)
    ax3.set_ylabel("MOS")

    # --- STD distributions
    ax4 = fig.add_subplot(gs[1, 1])
    if "mos_q_std" in df.columns:
        ax4.hist(df["mos_q_std"].dropna(), bins=30, color=BLUE,
                 alpha=0.6, label="Quality STD")
    if "mos_a_std" in df.columns:
        ax4.hist(df["mos_a_std"].dropna(), bins=30, color=PURPLE,
                 alpha=0.6, label="Alignment STD")
    if "mos_q_std" in df.columns or "mos_a_std" in df.columns:
        ax4.legend(fontsize=8)
        ax4.set_title("Score Std Dev Distributions", fontsize=11)
        ax4.set_xlabel("STD"); ax4.set_ylabel("Count")
    else:
        # Fallback: show mos_c distribution
        if len(mos_c):
            ax4.hist(mos_c, bins=30, color=CYAN, alpha=0.8)
            ax4.set_title("Aesthetic MOS Distribution (mos_c)", fontsize=11)
            ax4.set_xlabel("MOS"); ax4.set_ylabel("Count")

    # --- Descriptive stats table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    mos_cols = [(c, df[c].dropna()) for c in ["mos_q","mos_a","mos_c"]
                if c in df.columns and df[c].notna().sum() > 0]
    tdata = [["Metric","Mean","Std","Min","Max","Skew"]]
    for name, s in mos_cols:
        tdata.append([name,
                      f"{s.mean():.3f}", f"{s.std():.3f}",
                      f"{s.min():.3f}",  f"{s.max():.3f}",
                      f"{s.skew():.3f}"])
    tbl = ax5.table(cellText=tdata[1:], colLabels=tdata[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax5.set_title("Descriptive Statistics", fontsize=11, pad=8)

    out = out_dir / "fig1_mos_overview.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig1] Saved → {out}")


# ── Fig 2: Per-Model MOS Analysis ────────────────────────────────────────────
def fig2_per_model(df: pd.DataFrame, out_dir: Path):
    if "model" not in df.columns:
        df = df.copy()
        df["model"] = df["image_name"].apply(infer_model)

    models = df["model"].unique()
    palette = [BLUE, PINK, PURPLE, CYAN, "#FF6B35", "#2ECC71"]
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(sorted(models))}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("PKU-AIGIQA-4K — Per-Model MOS Analysis",
                 fontsize=15, fontweight="bold")

    for col_idx, (mos_col, mos_label) in enumerate([
            ("mos_q", "Quality MOS"), ("mos_a", "Alignment MOS")]):
        if mos_col not in df.columns:
            continue
        data = df[[mos_col, "model"]].dropna()
        groups = data.groupby("model")[mos_col]
        model_names = sorted(groups.groups.keys())
        means = [groups.get_group(m).mean() for m in model_names]
        stds  = [groups.get_group(m).std()  for m in model_names]
        colors = [color_map[m] for m in model_names]

        # Bar chart
        ax = axes[0, col_idx]
        bars = ax.bar(model_names, means, yerr=stds,
                      color=colors, alpha=0.85, capsize=5,
                      error_kw={"linewidth": 1.5})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"Mean {mos_label} by Model", fontsize=11)
        ax.set_ylabel(f"Mean MOS ± 1 SD"); ax.set_ylim(0, None)
        ax.tick_params(axis="x", rotation=20)

        # Violin
        ax = axes[1, col_idx]
        vdata = [groups.get_group(m).values for m in model_names]
        vp = ax.violinplot(vdata, positions=range(len(model_names)),
                           showmedians=True, showextrema=True)
        for i, (pc, m) in enumerate(zip(vp["bodies"], model_names)):
            pc.set_facecolor(color_map[m]); pc.set_alpha(0.7)
        vp["cmedians"].set_color("cyan"); vp["cmedians"].set_linewidth(2)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20)
        ax.set_title(f"{mos_label} Violin by Model", fontsize=11)
        ax.set_ylabel(mos_label)

    fig.tight_layout()
    out = out_dir / "fig2_per_model.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig2] Saved → {out}")


# ── Fig 3: Style & Adjective Analysis ────────────────────────────────────────
def fig3_style_adj(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    prompt_col = "image_prompt" if "image_prompt" in df.columns else \
                 "text_prompt"  if "text_prompt"  in df.columns else None

    if "style" not in df.columns:
        df["style"] = df[prompt_col].apply(infer_style) if prompt_col else "unspecified"

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("PKU-AIGIQA-4K — Style & Adjective Analysis",
                 fontsize=15, fontweight="bold")

    style_palette = [CYAN, PURPLE, "#7B2D8B", PINK, BLUE,
                     "#FF6B35", "#2ECC71", "#F39C12"]

    for col_idx, (mos_col, mos_label) in enumerate([
            ("mos_q", "Quality MOS"), ("mos_a", "Alignment MOS")]):
        if mos_col not in df.columns:
            continue
        style_means = (df.groupby("style")[mos_col].mean()
                         .sort_values().dropna())
        styles = style_means.index.tolist()
        colors = [style_palette[i % len(style_palette)] for i in range(len(styles))]

        # Horizontal bar
        ax = axes[0, col_idx]
        ax.barh(styles, style_means.values, color=colors, alpha=0.85)
        for i, v in enumerate(style_means.values):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{mos_label} by Style", fontsize=11)
        ax.set_xlabel(f"Mean {mos_label}")

        # Top adjectives / phrases
        ax = axes[1, col_idx]
        if prompt_col:
            # For each phrase, compute mean MOS
            all_phrases = {}
            for _, row in df[[prompt_col, mos_col]].dropna().iterrows():
                words = str(row[prompt_col]).lower().split()
                for i in range(len(words)-1):
                    phrase = f"{words[i]} {words[i+1]}"
                    if phrase not in all_phrases:
                        all_phrases[phrase] = []
                    all_phrases[phrase].append(row[mos_col])
            # Filter: at least 5 occurrences
            phrase_means = {p: np.mean(v) for p, v in all_phrases.items()
                            if len(v) >= 5}
            if phrase_means:
                top15 = sorted(phrase_means.items(), key=lambda x: x[1],
                                reverse=True)[:15]
                phrases, vals = zip(*top15)
                ax.barh(list(phrases), list(vals), color=PURPLE, alpha=0.85)
                for i, v in enumerate(vals):
                    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
                ax.set_title(f"Top 15 Phrases — {mos_label}", fontsize=11)
                ax.set_xlabel(f"Mean {mos_label}")
            else:
                ax.text(0.5, 0.5, "Insufficient phrase data",
                        transform=ax.transAxes, ha="center")
        else:
            ax.text(0.5, 0.5, "No prompt column found",
                    transform=ax.transAxes, ha="center")
        ax.set_title(f"Top Adjectives — {mos_label}", fontsize=11)

    fig.tight_layout()
    out = out_dir / "fig3_style_adj.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig3] Saved → {out}")


# ── Fig 4: Model × Style Heatmap ─────────────────────────────────────────────
def fig4_heatmap(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    prompt_col = "image_prompt" if "image_prompt" in df.columns else \
                 "text_prompt"  if "text_prompt"  in df.columns else None
    if "model" not in df.columns:
        df["model"] = df["image_name"].apply(infer_model)
    if "style" not in df.columns:
        df["style"] = df[prompt_col].apply(infer_style) if prompt_col else "unspecified"

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Mean MOS Heatmap: Model × Style",
                 fontsize=14, fontweight="bold")

    for ax, (mos_col, mos_label, cmap) in zip(axes, [
            ("mos_q", "Quality MOS",   "Blues"),
            ("mos_a", "Alignment MOS", "Purples")]):
        if mos_col not in df.columns:
            ax.text(0.5, 0.5, f"No {mos_col} column",
                    transform=ax.transAxes, ha="center")
            continue
        pivot = df.pivot_table(values=mos_col,
                               index="model", columns="style",
                               aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto",
                       cmap=cmap, vmin=pivot.values.min(),
                       vmax=pivot.values.max())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Style"); ax.set_ylabel("Model")
        ax.set_title(mos_label, fontsize=11)
        plt.colorbar(im, ax=ax, label="Mean MOS", shrink=0.8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if v > pivot.values.mean() else "black")

    fig.tight_layout()
    out = out_dir / "fig4_heatmap.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig4] Saved → {out}")


# ── Fig 5: Score Reliability & Inter-metric Correlations ─────────────────────
def fig5_reliability(df: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Score Reliability & Inter-metric Correlations",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Infer model for coloring
    df = df.copy()
    if "model" not in df.columns:
        df["model"] = df["image_name"].apply(infer_model)
    models = df["model"].unique()
    palette = [BLUE, PINK, PURPLE, CYAN, "#FF6B35", "#2ECC71"]
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(sorted(models))}
    point_colors = df["model"].map(color_map)

    # Scatter: mos_q vs mos_a STD (use mos_a as proxy for disagreement if no STD col)
    for ax_idx, (x_col, y_col, x_lbl, y_lbl, title) in enumerate([
            ("mos_q", "mos_a", "Quality MOS", "Alignment MOS", "Quality vs Alignment"),
            ("mos_a", "mos_c", "Alignment MOS", "Aesthetic MOS", "Alignment vs Aesthetic"),
    ]):
        ax = fig.add_subplot(gs[0, ax_idx])
        if x_col in df.columns and y_col in df.columns:
            sub = df[[x_col, y_col]].dropna()
            colors = point_colors.loc[sub.index] if len(point_colors) == len(df) \
                     else BLUE
            ax.scatter(sub[x_col], sub[y_col],
                       c=colors, alpha=0.35, s=6)
        ax.set_xlabel(x_lbl); ax.set_ylabel(y_lbl)
        ax.set_title(title, fontsize=10)

    # Correlation matrix
    ax = fig.add_subplot(gs[0, 2])
    mos_cols = [c for c in ["mos_q","mos_a","mos_c"] if c in df.columns]
    if len(mos_cols) >= 2:
        corr_data = df[mos_cols].dropna()
        corr_matrix = corr_data.corr().values
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(mos_cols)))
        ax.set_yticks(range(len(mos_cols)))
        ax.set_xticklabels(mos_cols, fontsize=9)
        ax.set_yticklabels(mos_cols, fontsize=9)
        ax.set_title("Score Correlation Matrix", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(mos_cols)):
            for j in range(len(mos_cols)):
                ax.text(j, i, f"{corr_matrix[i,j]:.2f}",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if abs(corr_matrix[i,j]) > 0.5 else "black")

    out = out_dir / "fig5_reliability.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig5] Saved → {out}")


# ── Fig 6: Sample Images (Low / Mid / High MOS) ───────────────────────────────
def fig6_samples(df: pd.DataFrame, img_dir: str, out_dir: Path,
                 n_per_tier: int = 3):
    img_dir = Path(img_dir)
    mos_q = df["mos_q"]
    q33, q66 = mos_q.quantile(0.33), mos_q.quantile(0.66)

    tiers = {
        "Low":  df[mos_q <= q33],
        "Mid":  df[(mos_q > q33) & (mos_q <= q66)],
        "High": df[mos_q > q66],
    }

    fig, axes = plt.subplots(3, n_per_tier, figsize=(5*n_per_tier, 5*3))
    fig.suptitle("Sample Images by mos_q (Low / Mid / High)",
                 fontsize=14, fontweight="bold")

    prompt_col = "image_prompt" if "image_prompt" in df.columns else \
                 "text_prompt"  if "text_prompt"  in df.columns else None

    for row_idx, (tier_name, tier_df) in enumerate(tiers.items()):
        samples = tier_df.sample(min(n_per_tier, len(tier_df)),
                                 random_state=42).reset_index(drop=True)
        for col_idx in range(n_per_tier):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(samples):
                continue
            row = samples.iloc[col_idx]
            img_path = img_dir / str(row.get("image_name", ""))
            try:
                img = PILImage.open(img_path).convert("RGB")
                ax.imshow(np.array(img))
            except Exception:
                ax.set_facecolor("black")
                ax.text(0.5, 0.5, "Image not found", transform=ax.transAxes,
                        ha="center", va="center", color="white", fontsize=8)

            prompt_snippet = ""
            if prompt_col and pd.notna(row.get(prompt_col, None)):
                prompt_snippet = str(row[prompt_col])[:40]

            ax.set_title(
                f"{tier_name} | mos_q={row['mos_q']:.3f}\n{prompt_snippet}",
                fontsize=8)

    fig.tight_layout()
    out = out_dir / "fig6_samples.png"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[fig6] Saved → {out}")


# ── Fig 7: Image Feature Analysis ────────────────────────────────────────────
def fig7_feature_analysis(feat_df: pd.DataFrame, out_dir: Path):
    if not HAS_SKLEARN:
        print("[fig7] Skipped — scikit-learn not available.")
        return
    if feat_df is None or len(feat_df) == 0:
        print("[fig7] Skipped — no feature data.")
        return

    feature_cols = [c for c in feat_df.columns
                    if c not in {"mos_q","mos_a","idx"}]
    X = feat_df[feature_cols].fillna(0).values
    y_q = feat_df["mos_q"].fillna(0).values
    y_a = feat_df["mos_a"].fillna(0).values if "mos_a" in feat_df.columns \
          else np.zeros(len(feat_df))

    # Pearson r for each feature
    corr_q = [abs(pearsonr(X[:,i], y_q)[0]) for i in range(X.shape[1])]
    corr_a = [pearsonr(X[:,i], y_a)[0]      for i in range(X.shape[1])]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Image Feature Analysis", fontsize=14, fontweight="bold")

    # Feature ↔ mos_q
    ax = axes[0, 0]
    order = np.argsort(corr_q)
    ax.barh([feature_cols[i] for i in order], [corr_q[i] for i in order],
            color=BLUE, alpha=0.85)
    ax.set_title("Feature ↔ Quality MOS", fontsize=11)
    ax.set_xlabel("Pearson r (absolute)")

    # Feature ↔ mos_a
    ax = axes[0, 1]
    order_a = np.argsort(corr_a)
    colors_a = [PURPLE if v >= 0 else PINK for v in [corr_a[i] for i in order_a]]
    ax.barh([feature_cols[i] for i in order_a],
            [corr_a[i] for i in order_a], color=colors_a, alpha=0.85)
    ax.set_title("Feature ↔ Alignment MOS", fontsize=11)
    ax.set_xlabel("Pearson r")

    # PCA coloured by mos_q
    ax = axes[1, 0]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=y_q,
                    cmap="viridis", alpha=0.5, s=10)
    plt.colorbar(sc, ax=ax, label="Quality MOS")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of Image Features (Quality MOS)", fontsize=11)

    # Best correlated feature scatter
    ax = axes[1, 1]
    best_idx = int(np.argmax(corr_q))
    best_feat = feature_cols[best_idx]
    r_val = pearsonr(X[:,best_idx], y_q)[0]
    ax.scatter(X[:,best_idx], y_q, alpha=0.35, s=8, color=BLUE)
    m, b = np.polyfit(X[:,best_idx], y_q, 1)
    xline = np.linspace(X[:,best_idx].min(), X[:,best_idx].max(), 100)
    ax.plot(xline, m*xline + b, color=PINK, lw=2, label=f"r = {r_val:.3f}")
    ax.set_xlabel(best_feat); ax.set_ylabel("Quality MOS")
    ax.set_title(f"Best Correlated Feature: {best_feat}", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = out_dir / "fig7_feature_analysis.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig7] Saved → {out}")


# ── Fig 8: ML Cross-Validation Results ───────────────────────────────────────
def fig8_ml_results(feat_df: pd.DataFrame, out_dir: Path):
    if not HAS_SKLEARN:
        print("[fig8] Skipped — scikit-learn not available.")
        return
    if feat_df is None or len(feat_df) == 0:
        print("[fig8] Skipped — no feature data.")
        return

    feature_cols = [c for c in feat_df.columns
                    if c not in {"mos_q","mos_a","idx"}]
    X = feat_df[feature_cols].fillna(0).values

    models_ml = {
        "Ridge":            Ridge(alpha=1.0),
        "Random Forest":    RandomForestRegressor(n_estimators=100,
                                                  random_state=42, n_jobs=-1),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=100,
                                                       random_state=42),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for target_name, target_col in [("Perception", "mos_q"),
                                      ("Alignment",  "mos_a")]:
        if target_col not in feat_df.columns:
            continue
        y = feat_df[target_col].fillna(0).values
        for model_name, estimator in models_ml.items():
            cv = cross_validate(estimator, X, y, cv=kf,
                                scoring=["neg_root_mean_squared_error",
                                         "r2"],
                                return_train_score=False)
            rmse = -cv["test_neg_root_mean_squared_error"].mean()
            r2   =  cv["test_r2"].mean()
            # SRCC
            from sklearn.model_selection import KFold as KF2
            srcc_scores = []
            for tr, te in KF2(n_splits=5, shuffle=True,
                              random_state=42).split(X):
                est = type(estimator)(**estimator.get_params())
                est.fit(X[tr], y[tr])
                preds = est.predict(X[te])
                srcc_scores.append(spearmanr(preds, y[te])[0])
            srcc = np.mean(srcc_scores)
            key = f"{model_name}\n({target_name})"
            results[key] = {"rmse": rmse, "r2": r2, "srcc": srcc,
                            "target": target_name, "model": model_name}

    if not results:
        print("[fig8] No results to plot.")
        return

    labels = list(results.keys())
    rmse_vals = [results[k]["rmse"] for k in labels]
    r2_vals   = [results[k]["r2"]   for k in labels]
    srcc_vals = [results[k]["srcc"] for k in labels]
    colors    = [BLUE if "Perception" in k else PURPLE for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Preliminary ML — 5-fold Cross-Validation",
                 fontsize=14, fontweight="bold")

    for ax, vals, title, better in zip(
            axes,
            [rmse_vals, r2_vals, srcc_vals],
            ["RMSE (↓ better)", "PLCC / R² (↑ better)", "SRCC (↑ better)"],
            ["lower", "higher", "higher"]):
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
        ax.set_title(title, fontsize=11)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=BLUE,   label="Perception"),
                            Patch(color=PURPLE, label="Alignment")],
                  fontsize=8, loc="upper right")

    fig.tight_layout()
    out = out_dir / "fig8_ml_results.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig8] Saved → {out}")


# ── Fig 9: Feature Importance ────────────────────────────────────────────────
def fig9_feature_importance(feat_df: pd.DataFrame, out_dir: Path):
    if not HAS_SKLEARN:
        print("[fig9] Skipped — scikit-learn not available.")
        return
    if feat_df is None or len(feat_df) == 0:
        print("[fig9] Skipped — no feature data.")
        return

    feature_cols = [c for c in feat_df.columns
                    if c not in {"mos_q","mos_a","idx"}]
    X = feat_df[feature_cols].fillna(0).values
    y = feat_df["mos_q"].fillna(0).values

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    order = np.argsort(importances)

    palette = [PINK, BLUE, "#2ECC71", "#FF6B35", CYAN, PURPLE,
               "#F39C12", "#E74C3C", "#1ABC9C", "#9B59B6",
               "#E67E22", "#95A5A6", "#34495E", "#16A085", "#C0392B",
               "#2980B9", "#8E44AD"]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [palette[i % len(palette)] for i in range(len(feature_cols))]
    ax.barh([feature_cols[i] for i in order],
            [importances[i] for i in order],
            color=[colors[i] for i in order], alpha=0.9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Random Forest Feature Importance (Quality MOS)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "fig9_feature_importance.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig9] Saved → {out}")


# =============================================================================
# SECTION 4 — MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate analysis figures for PKU-AIGIQA-4K")
    p.add_argument("--xlsx",    type=str, default=None,
                   help="Path to annotation.xlsx")
    p.add_argument("--csv",     type=str, default=None,
                   help="Path to annotations.csv")
    p.add_argument("--img-dir", type=str,
                   default="pku_aigiqa_4k/images",
                   help="Directory containing the images")
    p.add_argument("--out-dir", type=str, default="figures",
                   help="Output directory for figures")
    p.add_argument("--max-images", type=int, default=1000,
                   help="Max images for feature extraction (default: 1000)")
    p.add_argument("--skip-features", action="store_true",
                   help="Skip feature extraction (figs 7/8/9)")
    p.add_argument("--skip-samples",  action="store_true",
                   help="Skip sample images figure (fig 6)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(xlsx_path=args.xlsx, csv_path=args.csv)

    # Add model column if not present
    if "model" not in df.columns and "image_name" in df.columns:
        df["model"] = df["image_name"].apply(infer_model)

    # Add style column
    prompt_col = "image_prompt" if "image_prompt" in df.columns else \
                 "text_prompt"  if "text_prompt"  in df.columns else None
    if "style" not in df.columns:
        df["style"] = df[prompt_col].apply(infer_style) if prompt_col \
                      else "unspecified"

    print(f"\n[main] Generating figures → {out_dir}/\n")

    fig1_mos_overview(df, out_dir)
    fig2_per_model(df, out_dir)
    fig3_style_adj(df, out_dir)
    fig4_heatmap(df, out_dir)
    fig5_reliability(df, out_dir)

    if not args.skip_samples:
        fig6_samples(df, args.img_dir, out_dir)
    else:
        print("[fig6] Skipped (--skip-samples).")

    # Feature extraction for figs 7/8/9
    if not args.skip_features and HAS_SKLEARN:
        feat_df = extract_image_features(df, args.img_dir,
                                         max_images=args.max_images)
        if feat_df is not None and len(feat_df) > 10:
            fig7_feature_analysis(feat_df, out_dir)
            fig8_ml_results(feat_df, out_dir)
            fig9_feature_importance(feat_df, out_dir)
        else:
            print("[fig7/8/9] Skipped — too few images extracted.")
    elif args.skip_features:
        print("[fig7/8/9] Skipped (--skip-features).")

    print(f"\n[main] All done. Figures saved to '{out_dir}/'")


if __name__ == "__main__":
    main()