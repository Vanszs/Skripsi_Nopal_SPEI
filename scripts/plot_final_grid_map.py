"""
Plot map-style visualization for final selected grid nodes per city.

Outputs:
- results/maps/final_grid_overview.png
- results/maps/final_grid_per_city.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.ingest import build_city_nodes, load_city_centers


ROOT = Path(__file__).resolve().parents[1]
CITY_CONFIG = ROOT / "data" / "config" / "city_centers.json"
SELECTION_PATH = ROOT / "data" / "processed" / "node_selection_v2.parquet"
OUT_DIR = ROOT / "results" / "maps"


def _build_candidates_df(city_config_path: Path) -> pd.DataFrame:
    city_centers = load_city_centers(str(city_config_path))
    rows = []
    for city_id, coords in sorted(city_centers.items()):
        nodes = build_city_nodes(city_id, coords["lat"], coords["lon"])
        for n in nodes:
            rows.append(
                {
                    "city_id": city_id,
                    "node_local_id": n["node_local_id"],
                    "raw_node_id": n["raw_node_id"],
                    "lat": n["lat"],
                    "lon": n["lon"],
                    "is_center": n["node_local_id"] == "n00",
                }
            )
    return pd.DataFrame(rows)


def _plot_overview(candidates: pd.DataFrame, selected: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    city_list = sorted(candidates["city_id"].unique().tolist())
    color_map = {city: colors[i % len(colors)] for i, city in enumerate(city_list)}

    for city in city_list:
        city_cand = candidates[candidates["city_id"] == city]
        city_sel = selected[selected["city_id"] == city]
        c = color_map[city]

        ax.scatter(
            city_cand["lon"],
            city_cand["lat"],
            s=52,
            marker="o",
            edgecolors=c,
            facecolors="none",
            linewidths=1.4,
            alpha=0.85,
        )
        ax.scatter(
            city_sel["lon"],
            city_sel["lat"],
            s=95,
            marker="*",
            c=c,
            edgecolors="black",
            linewidths=0.5,
            label=f"{city} (selected=5)",
            zorder=5,
        )
        center = city_cand[city_cand["node_local_id"] == "n00"].iloc[0]
        ax.scatter(center["lon"], center["lat"], s=80, marker="x", c="black", zorder=6)
        ax.text(center["lon"] + 0.01, center["lat"] + 0.01, city, fontsize=9)

    ax.set_title("Final Grid Selection (5 Nodes per City)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_per_city(candidates: pd.DataFrame, selected: pd.DataFrame, out_path: Path):
    city_list = sorted(candidates["city_id"].unique().tolist())
    n = len(city_list)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.2 * nrows), squeeze=False)

    for idx, city in enumerate(city_list):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        city_cand = candidates[candidates["city_id"] == city]
        city_sel = selected[selected["city_id"] == city]
        sel_rank = {
            row["raw_node_id"]: int(row["selected_rank"]) for _, row in city_sel.iterrows()
        }

        ax.scatter(
            city_cand["lon"],
            city_cand["lat"],
            s=80,
            marker="o",
            edgecolors="tab:gray",
            facecolors="none",
            linewidths=1.2,
            alpha=0.9,
            label="candidate (9)",
        )

        ax.scatter(
            city_sel["lon"],
            city_sel["lat"],
            s=125,
            marker="*",
            c="tab:red",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="selected (5)",
        )

        center = city_cand[city_cand["node_local_id"] == "n00"].iloc[0]
        ax.scatter(center["lon"], center["lat"], s=90, marker="x", c="black", zorder=6, label="city center")

        for _, row in city_cand.iterrows():
            rid = row["raw_node_id"]
            if rid in sel_rank:
                label = f"r{sel_rank[rid]}:{row['node_local_id']}"
                ax.text(row["lon"] + 0.004, row["lat"] + 0.004, label, fontsize=8, color="tab:red")
            else:
                ax.text(row["lon"] + 0.004, row["lat"] + 0.004, row["node_local_id"], fontsize=7, color="tab:gray")

        ax.set_title(f"{city}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8)

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    fig.suptitle("Per-City Grid (9 candidates -> 5 selected)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = _build_candidates_df(CITY_CONFIG)
    selected = pd.read_parquet(SELECTION_PATH).copy()
    selected = selected[selected["selected_flag"] == True].copy()  # noqa: E712
    selected = selected.sort_values(["city_id", "selected_rank", "raw_node_id"])

    _plot_overview(candidates, selected, OUT_DIR / "final_grid_overview.png")
    _plot_per_city(candidates, selected, OUT_DIR / "final_grid_per_city.png")

    print(f"Saved: {OUT_DIR / 'final_grid_overview.png'}")
    print(f"Saved: {OUT_DIR / 'final_grid_per_city.png'}")


if __name__ == "__main__":
    main()
