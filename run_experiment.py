"""
run_experiment.py
=================
Single-command experiment runner:
  1. Wipes stale results/
  2. Trains TFT with specified encoder_length
  3. Runs full_evaluation.py on the new checkpoint
  4. Writes a Markdown report to results/

Usage:
    python run_experiment.py --encoder 30
    python run_experiment.py --encoder 90
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
CKPT_DIR    = ROOT / "logs" / "checkpoints"
PYTHON      = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str):
    bar = "=" * 68
    print(f"\n{bar}\n  {msg}\n{bar}")


def _run(cmd: list, label: str):
    """Run a subprocess and stream stdout/stderr in real time."""
    _banner(label)
    start = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT)
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    elapsed = time.time() - start
    print(f"\n[{'OK' if proc.returncode == 0 else 'FAILED'}] {label} ? {elapsed:.0f}s")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} exited with code {proc.returncode}")
    return "".join(lines)


def _best_checkpoint(enc: int) -> Path:
    """Return the best checkpoint matching enc{N}-*."""
    pattern = f"enc{enc}-*.ckpt"
    ckpts = sorted(CKPT_DIR.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint matching '{pattern}' in {CKPT_DIR}"
        )
    # lowest val_loss wins
    scored = []
    for p in ckpts:
        m = re.search(r"val_loss=([\d.]+)", p.name)
        if m:
            scored.append((float(m.group(1)), p))
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
    return ckpts[-1]


def _latest_eval_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob("full_eval_*"), reverse=True)
    if not dirs:
        raise FileNotFoundError("No full_eval_* directory found in results/")
    return dirs[0]


def _load_metrics(eval_dir: Path) -> dict:
    p = eval_dir / "metrics_summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _write_md_report(metrics: dict, eval_dir: Path, enc: int, elapsed_train: float):
    ov    = metrics.get("overall",   {})
    naive = metrics.get("naive_persistence", {})
    picp  = metrics.get("picp_overall", None)
    per_loc   = metrics.get("per_location",  {})
    per_horiz = metrics.get("per_horizon",   [])

    model_rmse = ov.get("rmse")
    naive_rmse = naive.get("rmse")
    skill = None
    if model_rmse and naive_rmse and naive_rmse > 0:
        skill = (1.0 - model_rmse / naive_rmse) * 100

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# TFT SPEI-3 Evaluation Report ? Encoder {enc}",
        f"",
        f"**Generated:** {ts}  ",
        f"**Checkpoint:** `{metrics.get('checkpoint', 'N/A')}`  ",
        f"**Encoder length:** {enc} hari  ",
        f"**Prediction length:** {metrics.get('prediction_length', 30)} hari  ",
        f"**Training time:** {elapsed_train:.0f}s ({elapsed_train/60:.1f} min)  ",
        f"**Train split:** {metrics.get('train_period', 'year < 2023')}  ",
        f"**Val split:** {metrics.get('val_period', 'year == 2023')}  ",
        f"**Test split:** {metrics.get('test_period', 'year >= 2024')}  ",
        f"",
        "---",
        "",
        "## Overall Metrics (Test Set ? 2024)",
        "",
        "| Metric | Model | Naive Persistence | Keterangan |",
        "|--------|-------|-------------------|------------|",
        f"| RMSE | **{model_rmse:.4f}** | {naive_rmse:.4f} | ? lebih baik |" if (model_rmse and naive_rmse) else "| RMSE | N/A | N/A | |",
        f"| MAE  | **{ov.get('mae', 0):.4f}** | {naive.get('mae', 0):.4f} | ? lebih baik |",
        f"| R2   | **{ov.get('r2', 0):.4f}** | {naive.get('r2', 0):.4f} | ? lebih baik |",
        f"| Pearson r | **{ov.get('pearson_r', 0):.4f}** | {naive.get('pearson_r', 0):.4f} | ? lebih baik |",
        f"| Bias | **{ov.get('bias', 0):.4f}** | {naive.get('bias', 0):.4f} | mendekati 0 lebih baik |",
        f"| PICP (P10?P90) | **{picp:.4f}** | ? | nominal = 0.80 |" if picp is not None else "| PICP | N/A | ? | |",
        f"| Skill Score | **{skill:.1f}%** | 0% | positif = model beats naive |" if skill is not None else "| Skill Score | N/A | | |",
        "",
        "> **PICP nominal:** P10?P90 interval diharapkan mencakup 80% data aktual.",
        "> **Skill Score:** `(1 - RMSE_model / RMSE_naive) ? 100%`",
        "",
        "---",
        "",
        "## Per-Location Metrics",
        "",
        "| Lokasi | RMSE | MAE | R2 | Pearson r | Bias | PICP |",
        "|--------|------|-----|----|-----------|------|------|",
    ]

    picp_per = metrics.get("picp_per_location", {})
    for loc, m in per_loc.items():
        pc = picp_per.get(loc, float("nan"))
        lines.append(
            f"| {loc} | {m.get('rmse',0):.4f} | {m.get('mae',0):.4f} | "
            f"{m.get('r2',0):.4f} | {m.get('pearson_r',0):.4f} | "
            f"{m.get('bias',0):.4f} | {pc:.4f} |"
        )

    # Horizon summary (day 1, 5, 10, 15, 20, 25, 30)
    highlight_days = {1, 5, 10, 15, 20, 25, 30}
    horiz_rows = [r for r in per_horiz if r.get("horizon") in highlight_days]

    if horiz_rows:
        lines += [
            "",
            "---",
            "",
            "## Degradasi Metrik per Horizon (hari ke-1, 5, 10, 15, 20, 25, 30)",
            "",
            "| Horizon | RMSE | MAE | Bias | Pearson r |",
            "|---------|------|-----|------|-----------|",
        ]
        for r in sorted(horiz_rows, key=lambda x: x["horizon"]):
            h    = r["horizon"]
            rmse = r.get("rmse")
            mae  = r.get("mae")
            bias = r.get("bias")
            corr = r.get("pearson_r")
            lines.append(
                f"| {h:>3} | "
                f"{rmse:.4f} | {mae:.4f} | {bias:.4f} | {corr:.4f} |"
                if all(v is not None for v in [rmse, mae, bias, corr])
                else f"| {h:>3} | N/A | N/A | N/A | N/A |"
            )

    lines += [
        "",
        "---",
        "",
        "## File Output",
        "",
        f"Semua hasil disimpan di: `{eval_dir}`",
        "",
        "| File | Keterangan |",
        "|------|------------|",
        "| `metrics_summary.json` | Semua metrik dalam format JSON |",
        "| `metrics_report.txt` | Laporan teks lengkap |",
        "| `predictions_full.csv` | Tabel prediksi vs aktual |",
        "| `horizon_metrics.csv` | RMSE/MAE/Bias per horizon step |",
        "| `classification_report.csv` | Akurasi per kelas SPEI (9-class) |",
        "| `classification_summary.csv` | Akurasi 3-class per lokasi |",
        "| `01_scatter_overall.png` | Scatter actual vs predicted |",
        "| `03_timeseries_per_location.png` | Time series overlay |",
        "| `06_horizon_metrics.png` | Degradasi metrik per horizon |",
        "| `08_quantile_fan.png` | Fan chart P10/P50/P90 |",
        "| `09_spei_classification.png` | Confusion matrix SPEI class |",
        "| `11_model_vs_naive_picp.png` | Skill vs naive + PICP coverage |",
        "",
        "---",
        "",
        "## Interpretasi",
        "",
    ]

    # Auto-interpretation hints
    if model_rmse and naive_rmse:
        if model_rmse < naive_rmse:
            lines.append(f"? **Model BEATS naive persistence** "
                         f"(RMSE {model_rmse:.4f} < {naive_rmse:.4f}).")
        else:
            lines.append(f"?? **Model TIDAK beats naive persistence** "
                         f"(RMSE {model_rmse:.4f} > {naive_rmse:.4f}). "
                         f"Perlu investigasi lebih lanjut.")

    r2_val = ov.get("r2", 0)
    if r2_val < 0:
        lines.append(f"?? **R2 = {r2_val:.4f} (negatif)** ? bias sistematis mendominasi "
                     f"variance unexplained. Cek bias = {ov.get('bias', 0):.4f}.")
    elif r2_val > 0.5:
        lines.append(f"? **R2 = {r2_val:.4f}** ? model menjelaskan lebih dari 50% variansi.")
    else:
        lines.append(f"?? **R2 = {r2_val:.4f}** ? menjelaskan <50% variansi; "
                     f"masih ada ruang improvement.")

    if picp is not None:
        if 0.75 <= picp <= 0.90:
            lines.append(f"? **PICP = {picp:.4f}** ? interval P10?P90 terkalibrasi baik "
                         f"(nominal 80%).")
        elif picp < 0.75:
            lines.append(f"?? **PICP = {picp:.4f}** ? under-coverage: interval terlalu sempit.")
        else:
            lines.append(f"?? **PICP = {picp:.4f}** ? over-coverage: interval konservatif.")

    lines.append("")

    report_path = eval_dir / f"REPORT_enc{enc}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report ? {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=int, default=30,
                        help="max_encoder_length for training (default: 30)")
    parser.add_argument("--epochs",  type=int, default=60,
                        help="max training epochs (default: 60)")
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, go straight to evaluation using best existing ckpt")
    args = parser.parse_args()

    enc = args.encoder

    # -- 1. Clean /results -------------------------------------------------
    _banner("STEP 1 ? Clearing stale results/")
    if RESULTS_DIR.exists():
        # Remove per-run eval directories only (keep static CSVs if needed)
        for d in RESULTS_DIR.glob("full_eval_*"):
            shutil.rmtree(d)
            print(f"  Removed {d.name}")
        # Remove loose files
        for f in RESULTS_DIR.glob("*.csv"):
            f.unlink(); print(f"  Removed {f.name}")
        for f in RESULTS_DIR.glob("*.json"):
            f.unlink(); print(f"  Removed {f.name}")
        for f in RESULTS_DIR.glob("*.txt"):
            f.unlink(); print(f"  Removed {f.name}")
        for f in RESULTS_DIR.glob("*.png"):
            f.unlink(); print(f"  Removed {f.name}")
    RESULTS_DIR.mkdir(exist_ok=True)
    print("  results/ cleared OK")

    # -- 2. Train -----------------------------------------------------------
    t_train_start = time.time()
    if not args.skip_train:
        _run(
            [PYTHON, "-c",
             f"import sys, os; sys.path.insert(0, os.getcwd()); "
             f"import torch; torch.set_float32_matmul_precision('medium'); "
             f"from src.training.train import train_pipeline; "
             f"print(train_pipeline(max_epochs={args.epochs}, "
             f"batch_size={args.batch}, max_encoder_length={enc}))"],
            f"STEP 2 ? Training  (encoder={enc}, max_epochs={args.epochs})"
        )
    else:
        print("  --skip-train: skipping training step.")
    elapsed_train = time.time() - t_train_start

    # -- 3. Resolve best checkpoint -----------------------------------------
    _banner("STEP 3 ? Resolving best checkpoint")
    ckpt = _best_checkpoint(enc)
    print(f"  Checkpoint : {ckpt}")

    # -- 4. Run full_evaluation.py ------------------------------------------
    _run(
        [PYTHON, "full_evaluation.py", "--checkpoint", str(ckpt)],
        f"STEP 4 ? Full evaluation"
    )

    # -- 5. Build MD report ------------------------------------------------
    _banner("STEP 5 ? Generating Markdown report")
    eval_dir = _latest_eval_dir()
    metrics  = _load_metrics(eval_dir)
    report   = _write_md_report(metrics, eval_dir, enc, elapsed_train)

    # -- 6. Print summary ---------------------------------------------------
    _banner("EXPERIMENT COMPLETE")
    ov    = metrics.get("overall", {})
    naive = metrics.get("naive_persistence", {})
    picp  = metrics.get("picp_overall", None)
    print(f"\n  Encoder length : {enc}")
    print(f"  Checkpoint     : {ckpt.name}")
    print(f"  Eval dir       : {eval_dir.name}")
    print(f"  Train time     : {elapsed_train:.0f}s")
    print()
    print("  --- MODEL METRICS ---")
    for k in ("rmse", "mae", "r2", "bias", "pearson_r"):
        v = ov.get(k)
        print(f"  {k.upper():<12}: {v:.4f}" if v is not None else f"  {k.upper():<12}: N/A")
    if picp is not None:
        print(f"  {'PICP':<12}: {picp:.4f}  (nominal 0.80)")
    print()
    print("  --- NAIVE BASELINE ---")
    for k in ("rmse", "mae", "r2"):
        v = naive.get(k)
        print(f"  {k.upper():<12}: {v:.4f}" if v is not None else f"  {k.upper():<12}: N/A")

    m_rmse = ov.get("rmse"); n_rmse = naive.get("rmse")
    if m_rmse and n_rmse and n_rmse > 0:
        skill = (1.0 - m_rmse / n_rmse) * 100
        verdict = "BEATS naive OK" if skill > 0 else "DOES NOT beat naive ?"
        print(f"\n  Skill Score  : {skill:.1f}%  ? Model {verdict}")

    print(f"\n  MD report    : {report}")


if __name__ == "__main__":
    main()
