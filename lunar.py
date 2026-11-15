import os
import math
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import plotly.graph_objects as go
from plotly.colors import get_colorscale

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
    "11": ("Apollo 11 - Tranquility Base (Mare Tranquillitatis)", 0.67416, 23.47314, 80),
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
    # Include coordinates in the cache key so different sites never collide
    output_file = os.path.join(
        output_dir, f"{site_slug}_{lat:+.2f}_{lon:+.2f}_{int(width_km)}km.tif"
    )

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

        # Snap to whole pixels so the saved transform matches the data read
        window = (
            from_bounds(min_x, min_y, max_x, max_y, src.transform)
            .round_offsets()
            .round_lengths()
        )
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


def render_to_html(tif_path: str, title: str, lat: float, lon: float) -> str:
    """Loads the local GeoTIFF and writes an interactive 3D render to an HTML file."""
    with rasterio.open(tif_path) as src:
        raw_dem = src.read(1).astype(np.float32)
        # Equirectangular grid: a pixel spans res·cos(lat) metres on the ground E-W
        res_x_m = abs(src.transform.a) * math.cos(math.radians(lat))
        res_y_m = abs(src.transform.e)
        nodata = src.nodata

        # Pixel position of the requested centre (lat, lon) inside the crop
        to_dem = Transformer.from_crs(
            "+proj=longlat +R=1737400 +no_defs", src.crs, always_xy=True
        )
        center_col, center_row = (~src.transform) * to_dem.transform(lon, lat)

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

    # Centre marker: ground position of (lat, lon) and the terrain height there.
    # The -0.5 moves from pixel edges to pixel centres; rows are counted from the
    # south because the DEM was flipped above.
    center_x_km = (center_col - 0.5) * (res_x_m / 1000.0)
    center_y_km = (rows - 1 - (center_row - 0.5)) * (res_y_m / 1000.0)
    ci = int(np.clip(round(center_col - 0.5), 0, cols - 1))
    cj = int(np.clip(round(rows - 1 - (center_row - 0.5)), 0, rows - 1))
    center_z_km = float(z_km[cj, ci]) if not np.isnan(z_km[cj, ci]) else 0.0
    center_elev_m = center_z_km * 1000.0 + min_elev_m
    pole_km = max(0.25 * relief_km, 0.5)  # height of the marker pole above the ground

    # plotly.js only knows a few colorscale names; pass explicit lists to buttons
    gray_scale = get_colorscale("Gray")
    turbo_scale = get_colorscale("Turbo")
    viridis_scale = get_colorscale("Viridis")
    # Ground extents for manual aspect ratios
    y_ratio = y_km[-1] / x_km[-1]

    print(
        f"Rendering 3D surface ({cols}x{rows} grid, max vertical relief: {relief_km:.2f} km)..."
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=x_km,
                y=y_km,
                z=z_km,
                colorscale=gray_scale,
                colorbar=dict(title="Elevation (km)", ticksuffix=" km"),
                # No highlight lines drawn across the surface while hovering
                contours=dict(
                    x=dict(highlight=False),
                    y=dict(highlight=False),
                    z=dict(highlight=False),
                ),
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
            ),
            # Vertical pole + marker showing the centre of the map
            go.Scatter3d(
                x=[center_x_km, center_x_km],
                y=[center_y_km, center_y_km],
                z=[center_z_km, center_z_km + pole_km],
                mode="lines+markers+text",
                line=dict(color="#ff3b30", width=6),
                marker=dict(size=[3, 7], color="#ff3b30", symbol=["circle", "diamond"]),
                text=["", "Center"],
                textposition="top center",
                textfont=dict(color="#ff3b30", size=13),
                name="Map center",
                showlegend=False,
                hovertemplate=(
                    f"Map center<br>Lat {lat:.3f}°, Lon {lon:.3f}°<br>"
                    f"Elevation: {center_elev_m:.0f} m<extra></extra>"
                ),
            ),
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
            # No projection lines from the cursor to the axis walls
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
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
                        args=[{"colorscale": [gray_scale]}, [0]],  # surface only
                    ),
                    dict(
                        label="Topo Heatmap",
                        method="restyle",
                        args=[{"colorscale": [turbo_scale]}, [0]],  # surface only
                    ),
                    dict(
                        label="Viridis",
                        method="restyle",
                        args=[{"colorscale": [viridis_scale]}, [0]],  # surface only
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
                                    x=1, y=y_ratio, z=(relief_km / (x_km[-1])) * 3.0
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
                                    x=1, y=y_ratio, z=(relief_km / (x_km[-1])) * 5.0
                                ),
                            }
                        ],
                    ),
                ],
            ),
        ],
    )

    html_path = os.path.splitext(tif_path)[0] + ".html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    size_mb = os.path.getsize(html_path) / (1024 * 1024)
    print(f"Saved interactive HTML: {html_path} ({size_mb:.1f} MB)")
    return html_path


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
    html_path = render_to_html(tif_path, name, lat, lon)
    print(f"Open it in a browser: {os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()