import requests
import pandas as pd
import os
import time
from datetime import datetime

# Central Rice Production Centers in East Java (Lat/Lon approximations for regency centers)
LOCATIONS = {
    "Lamongan":   {"lat": -7.128,  "lon": 112.316},
    "Ngawi":      {"lat": -7.403,  "lon": 111.445},
    "Bojonegoro": {"lat": -7.155,  "lon": 111.880},
    "Tuban":      {"lat": -6.895,  "lon": 112.045},
    "Nganjuk":    {"lat": -7.604,  "lon": 111.905}
}

START_DATE = "2005-01-01"
END_DATE = "2026-01-01"  # Exclusive end; fetches data through 2025-12-31

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

REQUIRED_VARIABLES = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm_mean",
    "temperature_2m_max",
    "temperature_2m_min"
]

def fetch_location_data(name, lat, lon, max_retries=3):
    # ... (omitted, params uses REQUIRED_VARIABLES joined)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(REQUIRED_VARIABLES),
        "timezone": "Asia/Bangkok"
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Fetching data for {name} (Attempt {attempt+1})...")
            response = requests.get(BASE_URL, params=params, timeout=20)
            
            if response.status_code != 200:
                print(f"FAILED {name}: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Extract daily data
            daily_data = data.get("daily")
            if not daily_data:
                print(f"WARNING: No daily data for {name}. Keys: {data.keys()}")
                continue

            
            try:
                # Explicit construction with RENAME on the fly
                df_data = {
                    "time": daily_data["time"],
                    "precipitation_sum": daily_data["precipitation_sum"],
                    "et0_fao_evapotranspiration": daily_data["et0_fao_evapotranspiration"],
                    "soil_moisture": daily_data["soil_moisture_0_to_7cm_mean"],
                    "temperature_2m_max": daily_data["temperature_2m_max"],
                    "temperature_2m_min": daily_data["temperature_2m_min"]
                }
                
                # Verify lengths before creation
                lengths = {k: len(v) for k, v in df_data.items()}
                if len(set(lengths.values())) > 1:
                     print(f"CRITICAL: Length mismatch for {name}: {lengths}")
                     # Try to trim to min length?
                     min_len = min(lengths.values())
                     for k in df_data:
                         df_data[k] = df_data[k][:min_len]
                     print(f"Trimmed {name} to {min_len} rows.")

                df = pd.DataFrame(df_data)
                df["location_id"] = name
                df["elevation"] = data.get("elevation", 0)
                
                return df
            except ValueError as ve:
                print(f"DataFrame Error for {name}: {ve}")
                print(f"Daily Data Keys: {daily_data.keys()}")
                print(f"Lengths: {[len(v) for k,v in daily_data.items() if isinstance(v, list)]}")
                raise ve
        
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            time.sleep(2 * (attempt + 1)) # Exponential backoff
            
    raise RuntimeError(f"Failed to fetch data for {name} after {max_retries} attempts")

def main():
    os.makedirs("data/raw", exist_ok=True)
    
    all_dfs = []
    
    for loc_name, coords in LOCATIONS.items():
        try:
            df = fetch_location_data(loc_name, coords["lat"], coords["lon"])
            all_dfs.append(df)
            print(f"Successfully fetched {len(df)} rows for {loc_name}")
            # Rate limiting: wait 3 seconds between requests to avoid 429 errors
            time.sleep(3)
        except Exception as e:
            print(f"CRITICAL ERROR: Could not fetch {loc_name}. Skipping. Details: {e}")
            continue
        
    if not all_dfs:
        raise RuntimeError("No data fetched for any location! Aborting.")
        
    # Combine all locations
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Validation
    print("\nData Validation:")
    print(full_df.info())
    print(full_df.groupby("location_id").count())
    
    output_path = "data/raw/weather_history_east_java.parquet"
    full_df.to_parquet(output_path, index=False)
    print(f"\nSaved raw dataset to {output_path}")

if __name__ == "__main__":
    main()
