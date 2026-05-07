"""
Dynamic visualization for predictions_full.csv (no fixed 5-location assumptions).
"""
import argparse
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _resolve_csv_path(csv_path_arg: str | None) -> Path:
    if csv_path_arg:
        p = Path(csv_path_arg)
        if not p.exists():
            raise FileNotFoundError(f"CSV path not found: {p}")
        return p
    candidates = sorted(Path("results").glob("full_eval_*/predictions_full.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No predictions_full.csv found under results/full_eval_*/. "
            "Run full_evaluation.py first or pass --csv."
        )
    return candidates[-1]

COLORS = {"actual": "#1f77b4", "pred": "#d62728", "band": "#f7b6b6"}

parser = argparse.ArgumentParser(description="Visualize predictions_full.csv")
parser.add_argument("--csv", type=str, default=None, help="Path to predictions_full.csv")
args = parser.parse_args()

CSV_PATH = _resolve_csv_path(args.csv)
OUT_DIR = CSV_PATH.parent / "actual_vs_predict"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, parse_dates=["time"])
entity_col = "super_node_id" if "super_node_id" in df.columns else "location_id"
group_col = "city_id" if "city_id" in df.columns else entity_col
groups = sorted(df[group_col].astype(str).unique().tolist())


def subplot_grid(n, max_cols=3):
    cols = min(max_cols, max(1, n))
    rows = math.ceil(n / cols)
    return rows, cols


def add_axes_grid(rows, cols, figsize):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    return fig, axes


# 1. Time series per group
rows, cols = subplot_grid(len(groups), max_cols=2)
fig, axes = add_axes_grid(rows, cols, figsize=(8 * cols, 4 * rows))
fig.suptitle(f"Actual vs Predicted SPEI-3 per {group_col} (Test 2024-2025)", fontsize=14, y=1.01)
for i, grp in enumerate(groups):
    ax = axes[i]
    sub = df[df[group_col].astype(str) == grp].sort_values("time")
    ax.fill_between(sub["time"], sub["pred_p10"], sub["pred_p90"], color=COLORS["band"], alpha=0.5, label="P10-P90")
    ax.plot(sub["time"], sub["actual"], color=COLORS["actual"], lw=1.4, label="Actual")
    ax.plot(sub["time"], sub["pred_p50"], color=COLORS["pred"], lw=1.0, ls="--", label="Pred P50")
    rmse = np.sqrt(np.mean(sub["error"] ** 2))
    r = np.corrcoef(sub["actual"], sub["pred_p50"])[0, 1]
    picp = sub["in_interval"].mean()
    ax.set_title(f"{grp}  RMSE={rmse:.3f}  r={r:.3f}  PICP={picp:.3f}", fontsize=10)
    ax.set_ylabel("SPEI-3")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
for j in range(len(groups), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "A1_timeseries_per_group.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Scatter per group + overall panel
n_panels = len(groups) + 1
rows, cols = subplot_grid(n_panels, max_cols=3)
fig, axes = add_axes_grid(rows, cols, figsize=(5 * cols, 4 * rows))
for i, grp in enumerate(groups):
    ax = axes[i]
    sub = df[df[group_col].astype(str) == grp]
    ax.scatter(sub["actual"], sub["pred_p50"], alpha=0.3, s=10, color=COLORS["pred"])
    lim = max(
        abs(sub["actual"].min()),
        abs(sub["actual"].max()),
        abs(sub["pred_p50"].min()),
        abs(sub["pred_p50"].max()),
    ) + 0.3
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    rmse = np.sqrt(np.mean(sub["error"] ** 2))
    r2 = 1 - np.sum(sub["error"] ** 2) / np.sum((sub["actual"] - sub["actual"].mean()) ** 2)
    r = np.corrcoef(sub["actual"], sub["pred_p50"])[0, 1]
    ax.set_title(f"{grp}\nRMSE={rmse:.3f} R2={r2:.3f} r={r:.3f}", fontsize=10)
    ax.set_xlabel("Actual SPEI-3")
    ax.set_ylabel("Predicted P50")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

ax = axes[len(groups)]
ax.scatter(df["actual"], df["pred_p50"], alpha=0.15, s=8, color="#555555")
lim = max(abs(df["actual"]).max(), abs(df["pred_p50"]).max()) + 0.3
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
rmse = np.sqrt(np.mean(df["error"] ** 2))
r2 = 1 - np.sum(df["error"] ** 2) / np.sum((df["actual"] - df["actual"].mean()) ** 2)
r = np.corrcoef(df["actual"], df["pred_p50"])[0, 1]
ax.set_title(f"OVERALL\nRMSE={rmse:.3f} R2={r2:.3f} r={r:.3f}", fontsize=10)
ax.set_xlabel("Actual SPEI-3")
ax.set_ylabel("Predicted P50")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

for j in range(n_panels, len(axes)):
    axes[j].axis("off")
fig.suptitle("Scatter Actual vs Predicted P50", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "A2_scatter_per_group.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Error distribution per group
rows, cols = subplot_grid(len(groups), max_cols=3)
fig, axes = add_axes_grid(rows, cols, figsize=(5 * cols, 3.8 * rows))
for i, grp in enumerate(groups):
    ax = axes[i]
    sub = df[df[group_col].astype(str) == grp]["error"]
    ax.hist(sub, bins=40, color=COLORS["pred"], alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(sub.mean(), color="orange", lw=1.5, ls="--", label=f"Bias={sub.mean():.3f}")
    ax.set_title(f"{grp}\nstd={sub.std():.3f}", fontsize=10)
    ax.legend(fontsize=8)
for j in range(len(groups), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "A3_error_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Confusion matrix per group
all_classes = sorted(df["actual_class"].unique().tolist())
rows, cols = subplot_grid(len(groups), max_cols=3)
fig, axes = add_axes_grid(rows, cols, figsize=(6 * cols, 5 * rows))
for i, grp in enumerate(groups):
    ax = axes[i]
    sub = df[df[group_col].astype(str) == grp]
    cm = confusion_matrix(sub["actual_class"], sub["pred_class"], labels=all_classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=all_classes)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(str(grp), fontsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(6)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(6)
for j in range(len(groups), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "A4_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. Monthly RMSE heatmap (month x group)
df["month_num"] = df["time"].dt.month
monthly_rmse = (
    df.groupby([group_col, "month_num"])["error"]
    .apply(lambda e: np.sqrt(np.mean(e**2)))
    .unstack(level=0)
    .sort_index()
)
month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
monthly_rmse = monthly_rmse.reindex(index=range(1, 13))
monthly_rmse.index = month_names

fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.2), 5))
im = ax.imshow(monthly_rmse.values.T, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(12))
ax.set_xticklabels(month_names)
ax.set_yticks(range(len(monthly_rmse.columns)))
ax.set_yticklabels([str(c) for c in monthly_rmse.columns])
plt.colorbar(im, ax=ax, label="RMSE")
ax.set_title(f"RMSE per Bulan x {group_col}", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "A5_monthly_rmse_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Monthly PICP
monthly_picp = df.groupby([group_col, "month_num"])["in_interval"].mean().unstack(level=0).reindex(index=range(1, 13))
monthly_picp.index = month_names
fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.2), 5))
for grp in groups:
    if grp in monthly_picp.columns:
        ax.plot(month_names, monthly_picp[grp], marker="o", label=grp)
ax.axhline(0.80, color="black", ls="--", lw=1, label="Nominal 80%")
ax.set_ylim(0, 1.05)
ax.set_ylabel("PICP")
ax.set_title(f"PICP Bulanan per {group_col}", fontsize=13)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(OUT_DIR / "A6_monthly_picp.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"All visualizations saved to: {OUT_DIR}")
