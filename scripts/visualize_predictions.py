"""
Visualisasi actual vs predicted SPEI-3
predictions_full.csv — enc30 final (hidden=48, dropout=0.35)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH  = "results/full_eval_20260303_191442/predictions_full.csv"
OUT_DIR   = Path("results/full_eval_20260303_191442/actual_vs_predict")
OUT_DIR.mkdir(exist_ok=True)

LOCATIONS = ["Bojonegoro", "Lamongan", "Nganjuk", "Ngawi", "Tuban"]
COLORS    = {"actual": "#1f77b4", "pred":   "#d62728", "band":  "#f7b6b6"}

SPEI_COLORS = {
    "Kekeringan Ekstrem":  "#8B0000",
    "Kekeringan Parah":    "#D2691E",
    "Kekeringan Sedang":   "#FFA500",
    "Kekeringan Ringan":   "#FFD700",
    "Normal":              "#90EE90",
    "Basah Ringan":        "#00BFFF",
    "Basah Sedang":        "#1E90FF",
    "Basah Parah":         "#00008B",
    "Basah Ekstrem":       "#4B0082",
}

df = pd.read_csv(CSV_PATH, parse_dates=["time"])
df["location_id"] = df["location_id"].astype(str)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Time series per lokasi (semua di 1 figure, 5 subplot)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=False)
fig.suptitle("Actual vs Predicted SPEI-3 per Kabupaten (Test 2024–2025)", fontsize=14, fontweight="bold", y=1.01)

for ax, loc in zip(axes, LOCATIONS):
    sub = df[df["location_id"] == loc].sort_values("time")
    ax.fill_between(sub["time"], sub["pred_p10"], sub["pred_p90"],
                    color=COLORS["band"], alpha=0.5, label="P10–P90")
    ax.plot(sub["time"], sub["actual"],   color=COLORS["actual"], lw=1.4, label="Actual")
    ax.plot(sub["time"], sub["pred_p50"], color=COLORS["pred"],   lw=1.0, ls="--", label="Pred P50")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axhline(-1, color="orange", lw=0.5, ls=":", alpha=0.7)
    ax.axhline( 1, color="steelblue", lw=0.5, ls=":", alpha=0.7)

    # Metrics per lokasi
    rmse = np.sqrt(np.mean(sub["error"]**2))
    r    = np.corrcoef(sub["actual"], sub["pred_p50"])[0,1]
    picp = sub["in_interval"].mean()
    ax.set_title(f"{loc}   RMSE={rmse:.3f}  r={r:.3f}  PICP={picp:.3f}", fontsize=10)
    ax.set_ylabel("SPEI-3")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_xlim(sub["time"].min(), sub["time"].max())

plt.tight_layout()
plt.savefig(OUT_DIR / "A1_timeseries_per_lokasi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: A1_timeseries_per_lokasi.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Scatter actual vs pred_p50 per lokasi (2×3 grid)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()

for i, loc in enumerate(LOCATIONS):
    ax = axes_flat[i]
    sub = df[df["location_id"] == loc]
    ax.scatter(sub["actual"], sub["pred_p50"], alpha=0.3, s=10, color=COLORS["pred"])
    lim = max(abs(sub["actual"].min()), abs(sub["actual"].max()),
              abs(sub["pred_p50"].min()), abs(sub["pred_p50"].max())) + 0.3
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y=x")
    rmse = np.sqrt(np.mean(sub["error"]**2))
    r2   = 1 - np.sum(sub["error"]**2) / np.sum((sub["actual"] - sub["actual"].mean())**2)
    r    = np.corrcoef(sub["actual"], sub["pred_p50"])[0,1]
    ax.set_title(f"{loc}\nRMSE={rmse:.3f}  R²={r2:.3f}  r={r:.3f}", fontsize=10)
    ax.set_xlabel("Actual SPEI-3")
    ax.set_ylabel("Predicted P50")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(fontsize=8)

# Overall scatter di subplot ke-6
ax = axes_flat[5]
ax.scatter(df["actual"], df["pred_p50"], alpha=0.15, s=8, color="#555555")
lim = max(abs(df["actual"].min()), abs(df["pred_p50"].min()),
          abs(df["actual"].max()), abs(df["pred_p50"].max())) + 0.3
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
rmse = np.sqrt(np.mean(df["error"]**2))
r2   = 1 - np.sum(df["error"]**2) / np.sum((df["actual"] - df["actual"].mean())**2)
r    = np.corrcoef(df["actual"], df["pred_p50"])[0,1]
ax.set_title(f"OVERALL\nRMSE={rmse:.3f}  R²={r2:.3f}  r={r:.3f}", fontsize=10)
ax.set_xlabel("Actual SPEI-3"); ax.set_ylabel("Predicted P50")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

fig.suptitle("Scatter Actual vs Predicted P50 — SPEI-3 (2024–2025)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "A2_scatter_per_lokasi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: A2_scatter_per_lokasi.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Error distribution per lokasi
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
for ax, loc in zip(axes, LOCATIONS):
    sub = df[df["location_id"] == loc]["error"]
    ax.hist(sub, bins=40, color=COLORS["pred"], alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(sub.mean(), color="orange", lw=1.5, ls="--", label=f"Bias={sub.mean():.3f}")
    ax.set_title(f"{loc}\nstd={sub.std():.3f}", fontsize=10)
    ax.set_xlabel("Error (actual − pred)")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Frekuensi")
fig.suptitle("Distribusi Error Prediksi SPEI-3 per Kabupaten", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "A3_error_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: A3_error_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SPEI class confusion: actual class vs pred class (heatmap per lokasi)
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Simpan semua classes yang muncul
all_classes = sorted(df["actual_class"].unique().tolist())

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for ax, loc in zip(axes, LOCATIONS):
    sub = df[df["location_id"] == loc]
    cm  = confusion_matrix(sub["actual_class"], sub["pred_class"], labels=all_classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=all_classes)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(loc, fontsize=10)
    ax.set_xlabel("Pred Class", fontsize=8)
    ax.set_ylabel("Actual Class", fontsize=8)
    for tick in ax.get_xticklabels(): tick.set_fontsize(6)
    for tick in ax.get_yticklabels(): tick.set_fontsize(6)

fig.suptitle("Confusion Matrix Kelas SPEI-3 per Kabupaten", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "A4_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: A4_confusion_matrix.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Monthly RMSE heatmap (bulan × lokasi)
# ═══════════════════════════════════════════════════════════════════════════════
df["month_num"] = df["time"].dt.month
monthly_rmse = (df.groupby(["location_id", "month_num"])["error"]
                  .apply(lambda e: np.sqrt(np.mean(e**2)))
                  .unstack(level=0))
month_names = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
monthly_rmse.index = [month_names[i-1] for i in monthly_rmse.index]

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(monthly_rmse.values[: , :].T, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(12)); ax.set_xticklabels(month_names)
ax.set_yticks(range(len(LOCATIONS))); ax.set_yticklabels([c for c in monthly_rmse.columns])
plt.colorbar(im, ax=ax, label="RMSE")
for i in range(len(monthly_rmse.columns)):
    for j in range(12):
        v = monthly_rmse.values[j, i]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > monthly_rmse.values.max()*0.6 else "black")
ax.set_title("RMSE per Bulan × Kabupaten", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "A5_monthly_rmse_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: A5_monthly_rmse_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PICP per bulan (apakah cakupan konsisten sepanjang tahun?)
# ═══════════════════════════════════════════════════════════════════════════════
monthly_picp = (df.groupby(["location_id", "month_num"])["in_interval"]
                  .mean().unstack(level=0))
monthly_picp.index = month_names

fig, ax = plt.subplots(figsize=(10, 5))
for loc in LOCATIONS:
    if loc in monthly_picp.columns:
        ax.plot(month_names, monthly_picp[loc], marker="o", label=loc)
ax.axhline(0.80, color="black", ls="--", lw=1, label="Nominal 80%")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Bulan"); ax.set_ylabel("PICP")
ax.set_title("PICP Bulanan per Kabupaten (P10–P90 Coverage)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "A6_monthly_picp.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: A6_monthly_picp.png")

print(f"\nSemua visualisasi tersimpan di: {OUT_DIR}")
