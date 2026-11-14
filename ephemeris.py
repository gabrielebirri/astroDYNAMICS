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



import numpy as np

def add_cartesian_vectors(df, prefix):
    """
    Converts Azimuth and Elevation (in degrees) to ENU Cartesian unit vectors.
    prefix: '_Sun' or '_Earth' based on the column names in the merged dataframe.
    """
    # Convert degrees to radians for numpy trigonometric functions
    az_rad = np.radians(df[f"Azimuth{prefix}"])
    el_rad = np.radians(df[f"Elevation{prefix}"])
    
    # Calculate ENU components
    df[f"X_East{prefix}"] = np.sin(az_rad) * np.cos(el_rad)
    df[f"Y_North{prefix}"] = np.cos(az_rad) * np.cos(el_rad)
    df[f"Z_Up{prefix}"] = np.sin(el_rad)
    
    return df

def generate_mission_timeline(lat: float, lon: float, alt: float, start_time: str, stop_time: str):
    """Fetches data, calculates vectors, and returns a JSON string for the frontend."""
    earth_data = get_lunar_ephemeris("399", lat, lon, alt, start_time, stop_time)
    sun_data = get_lunar_ephemeris("10", lat, lon, alt, start_time, stop_time)
    
    mission_timeline = pd.merge(sun_data, earth_data, on="Date_Time", suffixes=('_Sun', '_Earth'))
    mission_timeline = add_cartesian_vectors(mission_timeline, "_Sun")
    mission_timeline = add_cartesian_vectors(mission_timeline, "_Earth")
    
    # Return as a JSON string directly to the caller (no file saving)
    return mission_timeline.to_json(orient="records", date_format="iso")

# --- Usage Example: Simulating a South Pole Mission ---
if __name__ == "__main__":
    # Test the unified function
    json_payload = generate_mission_timeline(-89.9, 0.0, 0.0, "2027-01-01", "2027-01-15")
    print(json_payload[:500]) # Print just the first 500 characters to verify
    
