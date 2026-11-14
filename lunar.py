import os
import math
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import plotly.graph_objects as go

# 1. Configure GDAL to stream ONLY the exact pixel window over HTTP
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "50000000"  # 50 MB cache

# Official USGS Astrogeology LRO LOLA Global DEM (118m resolution)
USGS_LOLA_118M_URL = (
    "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
)

# 2. Catalog of Famous Lunar Sites: (Name, Center Lat, Center Lon, Box Width in km)
LUNAR_SITES = {
    "1": ("Tycho Crater (85 km crater with sharp central peak)", -43.31, -11.36, 130),
    "2": ("Copernicus Crater (93 km terraced impact crater)", 9.62, -20.08, 140),
    "3": (
        "Aristarchus & Vallis Schroteri (Crater + giant lava canyon)",
        24.5,
        -49.0,
        140,
    ),
    "4": ("Vallis Alpes (166 km valley slicing through Lunar Alps)", 49.2, 3.2, 180),
    "5": ("Apollo 15 - Mons Hadley & Hadley Rille", 26.13, 3.63, 90),
    "6": ("Apollo 17 - Taurus-Littrow Valley", 20.19, 30.77, 80),
    "7": ("Plato Crater (Flat-floored dark lava crater)", 51.6, -9.3, 140),
    "8": ("Montes Apenninus (5 km tall lunar mountain range)", 19.5, -1.5, 220),
    "9": ("Rupes Recta / Straight Wall (110 km linear fault cliff)", -21.8, -7.8, 110),
    "10": (
        "Tsiolkovskiy Crater (Far-side crater with central peak)",
        -20.4,
        129.1,
        220,
    ),
}


def get_bbox_from_center(lat: float, lon: float, width_km: float) -> tuple:
    """Converts center (lat, lon) and width in km into (min_lon, min_lat, max_lon, max_lat)."""
    km_per_deg_lat = 30.323  # 1 degree of lunar latitude ≈ 30.323 km
    half_lat = (width_km / 2.0) / km_per_deg_lat
    # Scale longitude by cos(lat) so the bounding box stays square in kilometers
    half_lon = half_lat / max(math.cos(math.radians(lat)), 0.1)
    return (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def fetch_lunar_dem(site_slug: str, lat: float, lon: float, width_km: float) -> str:
    """Streams the bounding box from USGS and saves a compressed local GeoTIFF."""
    output_dir = "lunar_dems"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{site_slug}_{int(width_km)}km.tif")

    # Skip network download if we already cached this exact file locally
    if os.path.exists(output_file):
        print(f"Found cached file: {output_file}")
        return output_file

    min_lon, min_lat, max_lon, max_lat = get_bbox_from_center(lat, lon, width_km)

    print(f"Connecting to USGS Astrogeology server...")
    with rasterio.open(USGS_LOLA_118M_URL) as src:
        lunar_crs = "+proj=longlat +R=1737400 +no_defs"
        transformer = Transformer.from_crs(lunar_crs, src.crs, always_xy=True)
        min_x, min_y = transformer.transform(min_lon, min_lat)
        max_x, max_y = transformer.transform(max_lon, max_lat)

        window = from_bounds(min_x, min_y, max_x, max_y, src.transform)
        w_px, h_px = int(window.width), int(window.height)
        print(
            f"Downloading window: {w_px} x {h_px} pixels (~{(w_px * h_px * 2) / 1e6:.1f} MB)..."
        )

        data = src.read(1, window=window)

        profile = src.profile.copy()
        profile.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": rasterio.windows.transform(window, src.transform),
                "compress": "lzw",
            }
        )

        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(data, 1)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved local GeoTIFF: {output_file} ({size_mb:.2f} MB)")
    return output_file


def render_in_plotly(tif_path: str, title: str, lat: float, lon: float):
    """Loads the local GeoTIFF and opens an interactive 3D render in the browser."""
    with rasterio.open(tif_path) as src:
        raw_dem = src.read(1).astype(np.float32)
        res_x_m = abs(src.transform.a)
        res_y_m = abs(src.transform.e)
        nodata = src.nodata

        # Check scale factor (LOLA 118m stores int16 in 0.5m units)
        scale_factor = src.scales[0] if src.scales and src.scales[0] != 1.0 else 0.5

    if nodata is not None:
        raw_dem[raw_dem == nodata] = np.nan

    # Flip vertically so North is at the top (+Y)
    dem_m = np.flipud(raw_dem) * scale_factor

    # Normalize elevation so the lowest point in the crop is 0.0 km
    min_elev_m = float(np.nanmin(dem_m))
    max_elev_m = float(np.nanmax(dem_m))
    relief_km = (max_elev_m - min_elev_m) / 1000.0
    z_km = (dem_m - min_elev_m) / 1000.0

    rows, cols = z_km.shape
    x_km = np.arange(cols) * (res_x_m / 1000.0)
    y_km = np.arange(rows) * (res_y_m / 1000.0)

    print(
        f"Rendering 3D surface ({cols}x{rows} grid, max vertical relief: {relief_km:.2f} km)..."
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_km,
                y=y_km,
                z=z_km,
                colorscale="Gray",
                colorbar=dict(title="Elevation (km)", ticksuffix=" km"),
                lighting=dict(
                    ambient=0.18,
                    diffuse=0.9,
                    roughness=0.85,
                    specular=0.05,
                    fresnel=0.1,
                ),
                lightposition=dict(x=1000, y=-1000, z=350),
                hovertemplate=(
                    "West-East: %{x:.1f} km<br>"
                    "South-North: %{y:.1f} km<br>"
                    "Relative Elevation: %{z:.2f} km<extra></extra>"
                ),
            )
        ]
    )

    # Add interactive UI buttons inside the browser for Color Map & Vertical Exaggeration
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:12px;color:#aaa;'>"
                f"Center: {lat:.2f}°, {lon:.2f}° | Total Vertical Relief: {relief_km:.2f} km"
                f"</span>"
            ),
            x=0.02,
            y=0.96,
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=60),
        scene=dict(
            xaxis_title="West → East (km)",
            yaxis_title="South → North (km)",
            zaxis_title="Elevation (km)",
            aspectmode="data",  # 1:1 true lunar scale by default
            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)),
        ),
        updatemenus=[
            # Dropdown 1: Color Scheme
            dict(
                type="buttons",
                direction="right",
                x=0.02,
                y=0.90,
                bgcolor="#222",
                font=dict(color="white", size=11),
                buttons=[
                    dict(
                        label="Lunar Gray",
                        method="restyle",
                        args=["colorscale", "Gray"],
                    ),
                    dict(
                        label="Topo Heatmap",
                        method="restyle",
                        args=["colorscale", "Turbo"],
                    ),
                    dict(
                        label="Viridis",
                        method="restyle",
                        args=["colorscale", "Viridis"],
                    ),
                ],
            ),
            # Dropdown 2: Vertical Scale (True 1x vs Exaggerated)
            dict(
                type="buttons",
                direction="right",
                x=0.30,
                y=0.90,
                bgcolor="#222",
                font=dict(color="white", size=11),
                buttons=[
                    dict(
                        label="1x True Scale",
                        method="relayout",
                        args=[{"scene.aspectmode": "data"}],
                    ),
                    dict(
                        label="3x Vertical Exaggeration",
                        method="relayout",
                        args=[
                            {
                                "scene.aspectmode": "manual",
                                "scene.aspectratio": dict(
                                    x=1, y=rows / cols, z=(relief_km / (x_km[-1])) * 3.0
                                ),
                            }
                        ],
                    ),
                    dict(
                        label="5x Vertical Exaggeration",
                        method="relayout",
                        args=[
                            {
                                "scene.aspectmode": "manual",
                                "scene.aspectratio": dict(
                                    x=1, y=rows / cols, z=(relief_km / (x_km[-1])) * 5.0
                                ),
                            }
                        ],
                    ),
                ],
            ),
        ],
    )

    fig.show()


def main():
    print("\n=== MOON TREK 3D TERRAIN EXPLORER ===")
    for key, (name, lat, lon, width) in LUNAR_SITES.items():
        print(f" [{key:>2}] {name} ({lat}°, {lon}° | {width} km box)")
    print(" [ C] Enter Custom Latitude / Longitude / Width")

    choice = (
        input("\nSelect a site number (default = 2 for Copernicus): ").strip().upper()
        or "2"
    )

    if choice == "C":
        name = (
            input("Enter site label (e.g., Kepler Crater): ").strip()
            or "Custom Lunar Site"
        )
        lat = float(input("Center Latitude (-60 to 60, e.g., 8.1): ").strip())
        lon = float(input("Center Longitude (-180 to 180, e.g., -38.0): ").strip())
        width_km = float(input("Box Width in km (e.g., 100): ").strip() or "100")
        slug = name.lower().replace(" ", "_")
    elif choice in LUNAR_SITES:
        name, lat, lon, width_km = LUNAR_SITES[choice]
        slug = name.split("(")[0].strip().lower().replace(" ", "_").replace("-", "")
    else:
        print("Invalid selection, defaulting to Copernicus Crater.")
        name, lat, lon, width_km = LUNAR_SITES["2"]
        slug = "copernicus_crater"

    tif_path = fetch_lunar_dem(slug, lat, lon, width_km)
    render_in_plotly(tif_path, name, lat, lon)


if __name__ == "__main__":
    main()
