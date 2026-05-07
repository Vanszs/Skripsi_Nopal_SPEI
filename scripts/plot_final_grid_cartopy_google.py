"""
Render final selected grid as square-only overlays on Google street tiles
using cartopy GoogleTiles(style='street').

Outputs:
- results/maps/final_grid_google_street_overview.png
- results/maps/final_grid_google_street_<city>.png
"""

from __future__ import annotations

import math
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")  # Set non-interactive backend to avoid Threading/Tkinter issues
import matplotlib.pyplot as plt
import pandas as pd
from cartopy.io.img_tiles import GoogleTiles
from shapely.geometry import box
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "data" / "processed" / "node_selection_v2.parquet"
OUT_DIR = ROOT / "results" / "maps"

# Grid cell side length (km). Aligned to ingest spacing (~0.12 deg lat ~13.3 km).
CELL_SIZE_KM = 13.3
LINE_COLOR = "#00C853"
FILL_COLOR = "#00E676"


def _cell_half_deltas_deg(lat: float, side_km: float) -> tuple[float, float]:
    half = side_km / 2.0
    dlat = half / 111.32
    dlon = half / (111.32 * max(0.2, math.cos(math.radians(lat))))
    return dlat, dlon


def _extent_for_points(df: pd.DataFrame, pad_deg: float = 0.15) -> tuple[float, float, float, float]:
    south = float(df["lat"].min()) - pad_deg
    north = float(df["lat"].max()) + pad_deg
    west = float(df["lon"].min()) - pad_deg
    east = float(df["lon"].max()) + pad_deg
    return (west, east, south, north)


def _draw_squares(ax, df: pd.DataFrame):
    # Create an outer boundary by unioning individual cell squares
    polygons = []
    for _, r in df.iterrows():
        lat = float(r["lat"])
        lon = float(r["lon"])
        dlat, dlon = _cell_half_deltas_deg(lat, CELL_SIZE_KM)
        west = lon - dlon
        south = lat - dlat
        polygons.append(box(west, south, west + 2.0 * dlon, south + 2.0 * dlat))
        
    super_node_shape = unary_union(polygons)
    
    ax.add_geometries(
        [super_node_shape],
        crs=ccrs.PlateCarree(),
        facecolor=FILL_COLOR,
        edgecolor=LINE_COLOR,
        alpha=0.2, # Slightly higher alpha for combined shape
        linewidth=2.5,
        zorder=6,
    )


def _render_map(df: pd.DataFrame, title: str, out_path: Path, zoom: int = 10):
    tiler = GoogleTiles(style="street")
    proj = tiler.crs
    fig = plt.figure(figsize=(12, 12), dpi=200)
    ax = plt.axes(projection=proj)
    ax.set_extent(_extent_for_points(df), crs=ccrs.PlateCarree())
    ax.add_image(tiler, zoom)
    _draw_squares(ax, df)

    ax.set_title(
        title + f"\n(square-only overlay, cell={CELL_SIZE_KM:.1f} km)",
        fontsize=14,
        pad=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SELECTION_PATH).copy()
    df = df[df["selected_flag"] == True].copy()  # noqa: E712
    df = df.sort_values(["city_id", "selected_rank", "raw_node_id"])

    # Overview across all selected squares.
    overview_out = OUT_DIR / "final_grid_google_street_overview.png"
    _render_map(df, "Final Grid Selection - All Cities", overview_out, zoom=11)
    print(f"Saved: {overview_out}")

    # Per-city maps.
    for city, g in df.groupby("city_id"):
        city_slug = str(city).lower().replace(" ", "_")
        city_out = OUT_DIR / f"final_grid_google_street_{city_slug}.png"
        _render_map(g, f"Final Grid Selection - {city}", city_out, zoom=13)
        print(f"Saved: {city_out}")


if __name__ == "__main__":
    main()
