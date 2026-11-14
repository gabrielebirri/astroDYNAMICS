import requests
import pandas as pd
from io import StringIO
from datetime import datetime

def get_lunar_ephemeris(target_id: str, lat: float, lon: float, alt: float, 
                        start_time: str, stop_time: str, step_size: str = "1 h") -> pd.DataFrame:
    """
    Fetches Azimuth and Elevation for a celestial body from a specific lunar surface coordinate.
    
    Parameters:
    - target_id: '10' for the Sun, '399' for Earth
    - lat, lon, alt: Selenographic coordinates (degrees) and altitude (km)
    - start_time, stop_time: ISO-8601 strings (e.g., '2027-01-01')
    - step_size: Temporal resolution (e.g., '1 h', '15 m')
    """
    url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    
    # Configure parameters for a topocentric observer on the Moon (Body 301)
    params = {
        "format": "text",
        "COMMAND": f"'{target_id}'",
        "OBJ_DATA": "NO",          # Suppress physical parameters block
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'coord@301'",   # Observer is attached to the Moon
        "COORD_TYPE": "GEODETIC",
        "SITE_COORD": f"'{lon},{lat},{alt}'", 
        "START_TIME": f"'{start_time}'",
        "STOP_TIME": f"'{stop_time}'",
        "STEP_SIZE": f"'{step_size}'",
        "QUANTITIES": "'4'",       # Request Apparent Azimuth & Elevation only
        "CSV_FORMAT": "YES"        # Output as comma-separated values
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    # Horizons wraps the requested data block between $$SOE and $$EOE markers
    text_data = response.text
    soe_idx = text_data.find("$$SOE")
    eoe_idx = text_data.find("$$EOE")
    
    if soe_idx == -1 or eoe_idx == -1:
        raise ValueError(f"Ephemeris block not found. API Response:\n{text_data[:500]}")
        
    csv_block = text_data[soe_idx + 5 : eoe_idx].strip()
    
    # Parse the CSV block. Horizons outputs columns: Date, Solar Presence, Lunar Presence, Azimuth, Elevation
    columns = ["Date_Time", "Solar_Presence", "Lunar_Presence", "Azimuth", "Elevation", "Empty"]
    
    df = pd.read_csv(StringIO(csv_block), names=columns, usecols=[0, 3, 4])
    
    # Clean up datetime strings and cast numerical columns
    df["Date_Time"] = df["Date_Time"].str.strip()
    df["Azimuth"] = pd.to_numeric(df["Azimuth"])
    df["Elevation"] = pd.to_numeric(df["Elevation"])
    
    return df

# --- Usage Example: Simulating a South Pole Mission ---
if __name__ == "__main__":
    # Target: Shackleton Crater Rim (approximate South Pole coordinates)
    target_lat = -89.9
    target_lon = 0.0
    target_alt = 0.0
    
    sim_start = "2027-01-01"
    sim_end = "2027-01-15" # Simulating a 14-day lunar surface window
    resolution = "1 h"
    
    print("Fetching Earth vectors...")
    earth_data = get_lunar_ephemeris("399", target_lat, target_lon, target_alt, sim_start, sim_end, resolution)
    
    print("Fetching Sun vectors...")
    sun_data = get_lunar_ephemeris("10", target_lat, target_lon, target_alt, sim_start, sim_end, resolution)
    
    # Merge datasets on the Date_Time column for a unified timeline
    mission_timeline = pd.merge(
        sun_data, 
        earth_data, 
        on="Date_Time", 
        suffixes=('_Sun', '_Earth')
    )
    
    # Convert to JSON for injection into a frontend framework
    json_output = mission_timeline.to_json(orient="records", date_format="iso")
    print(mission_timeline.head())
