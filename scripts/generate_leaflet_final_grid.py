"""
Generate Leaflet HTML map for final selected grids (square-only layer).

Output:
- results/maps/final_grid_leaflet.html
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "data" / "processed" / "node_selection_v2.parquet"
OUT_DIR = ROOT / "results" / "maps"
OUT_HTML = OUT_DIR / "final_grid_leaflet.html"

# Square size in kilometers for each final node cell.
CELL_SIZE_KM = 8.0


def _square_bounds(lat: float, lon: float, side_km: float) -> list[list[float]]:
    """
    Return Leaflet rectangle bounds [[south, west], [north, east]]
    using metric-square approximation around the center point.
    """
    half_km = side_km / 2.0
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * max(0.2, math.cos(math.radians(lat))))
    south = lat - dlat
    north = lat + dlat
    west = lon - dlon
    east = lon + dlon
    return [[south, west], [north, east]]


def _build_payload(df: pd.DataFrame) -> dict:
    city_colors = {
        "Bojonegoro": "#1f77b4",
        "Lamongan": "#ff7f0e",
        "Nganjuk": "#2ca02c",
        "Ngawi": "#d62728",
        "Tuban": "#9467bd",
    }

    rows = []
    for _, r in df.iterrows():
        city = str(r["city_id"])
        rows.append(
            {
                "city_id": city,
                "raw_node_id": str(r["raw_node_id"]),
                "selected_rank": int(r["selected_rank"]),
                "center": [float(r["lat"]), float(r["lon"])],
                "bounds": _square_bounds(float(r["lat"]), float(r["lon"]), CELL_SIZE_KM),
                "color": city_colors.get(city, "#00bcd4"),
            }
        )

    centers = [[float(r["lat"]), float(r["lon"])] for _, r in df.iterrows()]
    center_lat = sum(c[0] for c in centers) / len(centers)
    center_lon = sum(c[1] for c in centers) / len(centers)

    return {
        "cell_size_km": CELL_SIZE_KM,
        "map_center": [center_lat, center_lon],
        "rows": rows,
    }


def _html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Final Grid Leaflet Map</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      background: #0f172a;
      color: #e2e8f0;
      font-family: Segoe UI, system-ui, sans-serif;
    }}
    #map {{
      width: 100%;
      height: 100%;
    }}
    .leaflet-control-attribution {{
      font-size: 10px;
    }}
    .map-title {{
      position: absolute;
      top: 10px;
      left: 50px;
      z-index: 1000;
      background: rgba(2, 6, 23, 0.85);
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 13px;
      line-height: 1.35;
    }}
  </style>
</head>
<body>
  <div class="map-title">
    Final Grid Selection (Square-Only)<br/>
    Cell size: {payload["cell_size_km"]:.1f} km
  </div>
  <div id="map"></div>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const payload = {payload_json};

    const map = L.map('map', {{
      zoomControl: true,
      preferCanvas: true
    }}).setView(payload.map_center, 9);

    // Visual style close to modern web-map look.
    const esriImagery = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{
        attribution: 'Tiles &copy; Esri'
      }}
    );

    const cartoLight = L.tileLayer(
      'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
      {{
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 20
      }}
    );

    const osmStd = L.tileLayer(
      'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
      {{
        attribution: '&copy; OpenStreetMap contributors'
      }}
    );

    esriImagery.addTo(map);

    // Data layer: square grids only (no markers, no polylines).
    const squares = [];
    payload.rows.forEach((r) => {{
      const rect = L.rectangle(r.bounds, {{
        color: r.color,
        weight: 2,
        fillColor: r.color,
        fillOpacity: 0.18,
        lineCap: 'square',
        lineJoin: 'miter'
      }});
      rect.bindTooltip(
        `${{r.city_id}} | rank=${{r.selected_rank}} | ${{r.raw_node_id}}`,
        {{sticky: true}}
      );
      rect.addTo(map);
      squares.push(rect);
    }});

    const fg = L.featureGroup(squares);
    map.fitBounds(fg.getBounds().pad(0.18));

    L.control.layers(
      {{
        'Esri Imagery': esriImagery,
        'Carto Light': cartoLight,
        'OpenStreetMap': osmStd
      }},
      {{}},
      {{collapsed: false}}
    ).addTo(map);
  </script>
</body>
</html>
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SELECTION_PATH).copy()
    df = df[df["selected_flag"] == True].copy()  # noqa: E712
    df = df.sort_values(["city_id", "selected_rank", "raw_node_id"])

    payload = _build_payload(df)
    OUT_HTML.write_text(_html(payload), encoding="utf-8")
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
