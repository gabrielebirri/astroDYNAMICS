# Moon View

Moon View is an interactive lunar exploration and landing-site comparison project. It combines lunar terrain with the changing positions of the Sun, Earth, and selected satellites to help you explore what a location might be like. It is a visual and educational tool, not a mission-planning system or a guarantee that a site is safe to land on.

## What you can do

- Explore a 3D view of the Moon and zoom in on a landing site.
- View local terrain and follow how sunlight, Earth, and a selected satellite are seen from the surface.
- Compare visibility and terrain-slope scores for one or two locations over a chosen period.
- Open a 360-degree horizon view or a ground-level street view for a site.
- Try the separate `lunar.py` and `soil_composition.py` scripts for terrain and example soil-composition maps. Soil compositions and deposits shown by `soil_composition.py` are hypothetical illustrations, not measured lunar data.

## Run Moon View

You need Python 3 and an internet connection. From the project folder, install the Python packages:

```bash
python -m pip install numpy rasterio pyproj plotly flask
```

Start the app:

```bash
python server.py
```

The app opens in your browser at <http://127.0.0.1:5050>. To start it without opening a browser automatically:

```bash
python server.py --no-browser
```

Choose a site or enter coordinates, then use the tabs and site controls to explore the terrain, visibility layers, and landing-site assessment. The first visit to an area may take longer because the app downloads elevation data; downloaded terrain is cached in `lunar_dems/` for later use.

## How it works

The app gets lunar elevation from the USGS Lunar Orbiter Laser Altimeter (LOLA) global map and uses JPL Horizons to find where the Sun, Earth, or a satellite appears in the lunar sky. It combines those positions with the local terrain to estimate shadows, line of sight, and the horizon.

The landing assessment summarizes how often each object is visible, how long the longest visibility and blackout periods are, and how much of the terrain is below a chosen slope limit. These are comparative estimates based on the selected time interval and map resolution, not certified landing-safety ratings. The leaderboard and example soil layers are illustrative rather than scientific measurements.

The main web app is served by `server.py`; `Lunar_Horizon_Simulator_Skeleton.html` provides its interface. Supporting scripts handle terrain and horizon views, sunlight, Earth visibility, satellite visibility, and ground-level views. Source credits are listed in [`sources.md`](sources.md) and on the app's Credits page.

## Sources

These are the sources credited by the project in [`sources.md`](sources.md):

- **Lunar ephemerides:** [JPL Horizons System](https://ssd.jpl.nasa.gov/horizons/)
- **Lunar topography:** [USGS Astrogeology Planetary Maps](https://planetarymaps.usgs.gov/) — including the global LOLA elevation map used by the app.
- **Notable lunar place names:** [USGS Gazetteer of Planetary Nomenclature](https://planetarynames.wr.usgs.gov/)
- **Apollo landing-site reference:** [LROC QuickMap](https://quickmap.lroc.asu.edu/)
- **Moon 3D visualization reference:** [NASA Eyes on the Solar System](https://eyes.nasa.gov/apps/solar-system/#/earth/moons/moon?embed=true)
- **Additional project reference:** [Space-Ros Curiosity rover demo](https://github.com/space-ros/demos/blob/main/curiosity_rover/)

The project also uses external browser resources for some page styling, fonts, and 3D rendering, so those parts may require an internet connection.
