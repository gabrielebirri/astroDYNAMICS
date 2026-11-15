"""Render lunar-orbiter visibility on lunar terrain using JPL Horizons and LOLA."""

import argparse
import csv
import math
import os
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
from pyproj import Transformer

from horizon import (
    LONLAT,
    R_LUNA,
    SITI,
    USGS_LOLA_118M_URL,
    base_locale,
    campiona,
    carica_ritaglio,
    scarica_ritaglio,
)

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
FILE_SATELLITI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "satellites_id.csv")
SCALA_VISIBILITA = [
    [0.0, "#111820"],
    [1.0 / 255.0, "#111820"],
    [1.0 / 255.0, "#0284c7"],
    [1.0, "#bae6fd"],
]


# --------------------------------------------------------------------------
# Effemeridi da JPL Horizons
# --------------------------------------------------------------------------
def get_lunar_ephemeris(target_id, lat, lon, alt_km, start_time, stop_time, step_size="1 h"):
    """
    Azimut ed elevazione di un corpo celeste visto da un punto della superficie lunare.

    target_id: identifier or name of a lunar-orbiting spacecraft in Horizons
    lat, lon: coordinate selenografiche in gradi (longitudine est)
    alt_km: quota sul raggio di riferimento, in km
    start_time, stop_time: es. '2027-01-01 00:00'
    step_size: es. '1 h', '30 m', '1 d'

    Restituisce (datetime UTC, azimut, elevazione, distanza in km).
    Azimut: 0 = nord, 90 = est.
    """
    params = {
        "format": "text",
        "COMMAND": f"'{target_id}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'coord@301'",  # osservatore sulla superficie della Luna
        "COORD_TYPE": "GEODETIC",
        "SITE_COORD": f"'{lon},{lat},{alt_km}'",  # longitudine est, latitudine, quota (km)
        "START_TIME": f"'{start_time}'",
        "STOP_TIME": f"'{stop_time}'",
        "STEP_SIZE": f"'{step_size}'",
        "QUANTITIES": "'4,20'",  # azimuth/elevation and observer-target range
        "RANGE_UNITS": "KM",
        "ANG_FORMAT": "DEG",
        "CSV_FORMAT": "YES",
    }
    url = HORIZONS_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as risposta:
        testo = risposta.read().decode("utf-8", errors="replace")

    # Horizons racchiude la tabella tra i marcatori $$SOE e $$EOE
    i_soe, i_eoe = testo.find("$$SOE"), testo.find("$$EOE")
    if i_soe == -1 or i_eoe == -1:
        raise ValueError(f"Ephemeris table not found. Horizons answered:\n{testo[:1500]}")

    righe = []
    # Columns: date, solar/lunar flags, azimuth, elevation, range, range rate.
    for riga in testo[i_soe + 5 : i_eoe].strip().splitlines():
        campi = [c.strip() for c in riga.split(",")]
        if len(campi) < 6:
            continue
        istante = datetime.strptime(campi[0], "%Y-%b-%d %H:%M").replace(tzinfo=timezone.utc)
        righe.append((istante, float(campi[3]), float(campi[4]), float(campi[5])))
    if not righe:
        raise ValueError("Horizons returned an empty table.")
    return righe


def direzione_enu(az_deg, el_deg):
    """Versore (est, nord, alto) che punta verso azimut ed elevazione dati."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])


# --------------------------------------------------------------------------
# Terreno su un piano tangente al sito
# --------------------------------------------------------------------------
def griglia_locale(dem, transform, crs, lat, lon, semilato_m, passo_m):
    """
    Ricampiona il DEM su una griglia regolare (est, nord) centrata sul sito.

    I punti sono la proiezione ortografica della sfera sul piano tangente al
    sito, quindi la griglia funziona anche ai poli. La quota Z e' misurata lungo
    la verticale del sito e include la curvatura della Luna: in questo sistema
    di riferimento e la posizione del satellite possono essere proiettate su
    questa griglia locale.

    Restituisce (coordinate in metri, Z in metri). Z vale NaN fuori dal DEM.
    """
    alto, est, nord = base_locale(lat, lon)
    n = int(round(semilato_m / passo_m))
    coord = np.arange(-n, n + 1) * passo_m
    e, nn = np.meshgrid(coord / R_LUNA, coord / R_LUNA)  # righe = nord, colonne = est
    w = np.sqrt(np.clip(1.0 - e**2 - nn**2, 0.0, None))
    u = w[..., None] * alto + e[..., None] * est + nn[..., None] * nord
    lat_p = np.degrees(np.arcsin(np.clip(u[..., 2], -1.0, 1.0)))
    lon_p = np.degrees(np.arctan2(u[..., 1], u[..., 0]))

    tr = Transformer.from_crs(LONLAT, crs, always_xy=True)
    x, y = tr.transform(lon_p, lat_p)
    quota = campiona(dem, transform, np.asarray(x), np.asarray(y))
    z = (R_LUNA + quota) * w - R_LUNA
    return coord, z


def campiona_griglia(z, passo_m, n_mezzo, e, n):
    """Quota bilineare della griglia locale nei punti (e, n) in metri. Fuori -> -inf."""
    col = e / passo_m + n_mezzo
    row = n / passo_m + n_mezzo
    righe, colonne = z.shape
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    valido = (c0 >= 0) & (c0 < colonne - 1) & (r0 >= 0) & (r0 < righe - 1)
    c0 = np.clip(c0, 0, colonne - 2)
    r0 = np.clip(r0, 0, righe - 2)
    fx, fy = col - c0, row - r0
    q = (
        z[r0, c0] * (1 - fx) * (1 - fy)
        + z[r0, c0 + 1] * fx * (1 - fy)
        + z[r0 + 1, c0] * (1 - fx) * fy
        + z[r0 + 1, c0 + 1] * fx * fy
    )
    q = np.where(valido, q, -np.inf)
    return np.where(np.isnan(q), -np.inf, q)


def distanze_di_marcia(passo_m, d_max):
    """Terrain samples are dense near the site and grow by 2% farther away."""
    d = list(np.arange(1, 41) * passo_m)
    while d[-1] < d_max:
        d.append(d[-1] + max(passo_m, 0.02 * d[-1]))
    return np.array(d)


# --------------------------------------------------------------------------
# Visibilita' del satellite
# --------------------------------------------------------------------------
def visibilita_satellite(
    z_ext, passo_m, n_ext, i0, i1, az_deg, el_deg, range_km, sito_alt_m, distanze
):
    """
    Point-source visibility from each displayed terrain point.

    The finite range gives each terrain point its own line of sight, including
    parallax. Rays are checked against both sampled terrain and the lunar body.
    """
    z0 = z_ext[i0:i1, i0:i1]
    coord = (np.arange(i0, i1) - n_ext) * passo_m
    e0, n0 = np.meshgrid(coord, coord)

    az, el = math.radians(az_deg), math.radians(el_deg)
    range_m = range_km * 1000.0
    satellite = np.array(
        [
            range_m * math.sin(az) * math.cos(el),
            range_m * math.cos(az) * math.cos(el),
            R_LUNA + sito_alt_m + range_m * math.sin(el),
        ]
    )
    point = np.stack(
        (e0, n0, np.full(z0.shape, R_LUNA) + z0), axis=-1
    )
    ray = satellite - point
    horizontal = np.hypot(ray[..., 0], ray[..., 1])
    elevation_slope = np.divide(
        ray[..., 2],
        horizontal,
        out=np.full(z0.shape, np.inf),
        where=horizontal > 0,
    )

    # A straight line through the Moon is occulted regardless of local relief.
    ray_length_sq = np.sum(ray * ray, axis=-1)
    closest_t = np.divide(
        -np.sum(point * ray, axis=-1),
        ray_length_sq,
        out=np.zeros(z0.shape),
        where=ray_length_sq > 0,
    )
    closest = point + np.clip(closest_t, 0.0, 1.0)[..., None] * ray
    occulted = (
        (closest_t > 0.0)
        & (closest_t < 1.0)
        & (np.sum(closest * closest, axis=-1) < R_LUNA**2)
    )

    direction_e = np.divide(
        ray[..., 0], horizontal, out=np.zeros(z0.shape), where=horizontal > 0
    )
    direction_n = np.divide(
        ray[..., 1], horizontal, out=np.zeros(z0.shape), where=horizontal > 0
    )
    blocked = occulted
    for d in distanze:
        if np.all(d >= horizontal):
            break
        zt = campiona_griglia(
            z_ext,
            passo_m,
            n_ext,
            e0 + d * direction_e,
            n0 + d * direction_n,
        )
        blocked |= (d < horizontal) & (zt > z0 + d * elevation_slope)
    return (~blocked).astype(float)


def in_scala_colori(visibilita):
    """Map satellite visibility to the 0..255 blue terrain colorscale."""
    v = np.where(visibilita > 0.5, 255.0, 0.0)
    return np.where(np.isnan(v), 0, v).astype(np.uint8)


# --------------------------------------------------------------------------
# Rendering 3D animato
# --------------------------------------------------------------------------
def disegna(nome, satellite, x_km, y_km, z_km, fotogrammi, file_html):
    """fotogrammi: (istante, azimut, elevazione, colori uint8, % terreno visibile)."""
    rilievo_km = float(np.nanmax(z_km))
    centro = len(x_km) // 2
    z_centro = float(np.nan_to_num(z_km[centro, centro]))
    lato_km = float(x_km[-1] - x_km[0])
    lunghezza_freccia = 0.35 * lato_km

    def freccia_satellite(az, el):
        s = direzione_enu(az, el) * lunghezza_freccia
        colore = "#38bdf8" if el > 0 else "#6c757d"
        return go.Scatter3d(
            x=[0.0, s[0]], y=[0.0, s[1]], z=[z_centro, z_centro + s[2]],
            mode="lines+markers+text",
            line=dict(color=colore, width=6),
            marker=dict(size=[3, 9], color=colore),
            text=["", satellite], textposition="top center", textfont=dict(color=colore, size=13),
            hovertemplate=f"{satellite}<br>Azimut {az:.1f}°<br>Elevazione {el:.2f}°<extra></extra>",
            showlegend=False,
        )

    def titolo(istante, az, el, perc):
        return (
            f"<b>Visibilità di {satellite} - {nome}</b><br><span style='font-size:12px;color:#aaa'>"
            f"{istante:%Y-%m-%d %H:%M} UTC | Azimut {az:.1f}°, elevazione {el:+.2f}° | "
            f"terreno con satellite visibile {perc:.1f}%</span>"
        )

    def superficie(colori):
        return go.Surface(
            x=x_km, y=y_km, z=z_km,
            surfacecolor=colori, cmin=0, cmax=255, colorscale=SCALA_VISIBILITA,
            colorbar=dict(
                title=dict(text="Satellite visibile", font=dict(size=12)), thickness=14, len=0.6,
                tickvals=[0, 255], ticktext=["nascosto", "visibile"],
            ),
            # Visibility is calculated from terrain horizons; Plotly adds relief shading.
            lighting=dict(ambient=0.8, diffuse=0.25, specular=0.0, roughness=1.0, fresnel=0.0),
            contours=dict(x=dict(highlight=False), y=dict(highlight=False), z=dict(highlight=False)),
            hovertemplate=(
                "Ovest-Est: %{x:.1f} km<br>Sud-Nord: %{y:.1f} km<br>"
                "Quota relativa: %{z:.2f} km<br>Satellite visibile: %{surfacecolor}<extra></extra>"
            ),
        )

    t0, az0, el0, col0, perc0 = fotogrammi[0]
    fig = go.Figure(data=[superficie(col0), freccia_satellite(az0, el0)])
    fig.frames = [
        go.Frame(
            name=str(k),
            data=[go.Surface(surfacecolor=col), freccia_satellite(az, el)],
            traces=[0, 1],
            layout=dict(title=dict(text=titolo(t, az, el, perc))),
        )
        for k, (t, az, el, col, perc) in enumerate(fotogrammi)
    ]

    animazione = dict(frame=dict(duration=350, redraw=True), transition=dict(duration=0), mode="immediate")
    cursore = dict(
        active=0, x=0.05, y=0.02, len=0.9, pad=dict(t=10, b=10),
        currentvalue=dict(prefix="Istante: ", font=dict(size=12)),
        steps=[
            dict(label=f"{t:%m-%d %H:%M}", method="animate",
                 args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
            for k, (t, *_resto) in enumerate(fotogrammi)
        ],
    )
    rapporto_z = rilievo_km / lato_km
    fig.update_layout(
        title=dict(text=titolo(t0, az0, el0, perc0), x=0.02, y=0.96),
        template="plotly_dark", paper_bgcolor="#05060a",
        margin=dict(l=0, r=0, b=0, t=70),
        scene=dict(
            xaxis_title="Ovest → Est (km)", yaxis_title="Sud → Nord (km)", zaxis_title="Quota (km)",
            xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False,
            aspectmode="data", camera=dict(eye=dict(x=1.2, y=-1.2, z=0.9)),
        ),
        sliders=[cursore],
        updatemenus=[
            dict(
                type="buttons", direction="right", x=0.02, y=0.90, showactive=False,
                bgcolor="#222", font=dict(color="white", size=11),
                buttons=[
                    dict(label="▶ Avvia", method="animate", args=[None, dict(animazione, fromcurrent=True)]),
                    dict(label="❚❚ Pausa", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            ),
            dict(
                type="buttons", direction="right", x=0.30, y=0.90,
                bgcolor="#222", font=dict(color="white", size=11),
                buttons=[
                    dict(label="Scala reale", method="relayout", args=[{"scene.aspectmode": "data"}]),
                    dict(label="Rilievo x3", method="relayout",
                         args=[{"scene.aspectmode": "manual",
                                "scene.aspectratio": dict(x=1, y=1, z=rapporto_z * 3.0)}]),
                    dict(label="Rilievo x5", method="relayout",
                         args=[{"scene.aspectmode": "manual",
                                "scene.aspectratio": dict(x=1, y=1, z=rapporto_z * 5.0)}]),
                ],
            ),
        ],
    )
    fig.write_html(file_html, include_plotlyjs=True, full_html=True, auto_play=False,
                   config={"displaylogo": False})
    return fig


# --------------------------------------------------------------------------
def scegli_sito(a):
    """Restituisce (nome, lat, lon) dagli argomenti o dal menu interattivo."""
    if a.lat is not None and a.lon is not None:
        return a.nome or "Custom site", a.lat, a.lon
    scelta = a.sito
    if scelta is None:
        print("\n=== VISIBILITA' DEI SATELLITI DALLA LUNA ===")
        for chiave, (nome, lat, lon) in SITI.items():
            print(f" [{chiave:>2}] {nome} ({lat}°, {lon}°)")
        print(" [ C] Latitudine e longitudine personalizzate")
        scelta = input("\nScegli un sito (predefinito = 1, Tycho): ").strip().upper() or "1"
    if scelta == "C":
        nome = input("Nome del sito: ").strip() or "Custom site"
        lat = float(input("Latitudine (-90 a 90): ").strip())
        lon = float(input("Longitudine est (-180 a 360): ").strip())
        return nome, lat, lon
    if scelta not in SITI:
        print("Scelta non valida, uso Tycho.")
        scelta = "1"
    return SITI[scelta]


def carica_catalogo(percorso=FILE_SATELLITI):
    """Legge satellites_id.csv (Nome;ID, senza intestazione) e restituisce {nome: id Horizons}."""
    catalogo = {}
    if not os.path.exists(percorso):
        return catalogo
    with open(percorso, newline="", encoding="utf-8-sig") as f:
        for riga in csv.reader(f, delimiter=";"):
            if len(riga) >= 2 and riga[0].strip() and riga[1].strip():
                catalogo[riga[0].strip()] = riga[1].strip()
    return catalogo


def risolvi_satellite(testo, catalogo=None):
    """
    Restituisce (nome, id Horizons) per un nome o un ID.

    I nomi del catalogo (senza distinzione tra maiuscole e minuscole) vengono
    tradotti nel loro ID; un ID del catalogo riprende il suo nome. Tutto il
    resto viene passato a Horizons cosi' com'e'.
    """
    catalogo = carica_catalogo() if catalogo is None else catalogo
    e = testo.strip().strip("\"'")
    for nome, id_horizons in catalogo.items():
        if e.lower() == nome.lower() or e == id_horizons:
            return nome, id_horizons
    return e, e


def scegli_satelliti(valore):
    """Restituisce [(nome, id Horizons)] da numeri di menu, nomi del catalogo o ID liberi."""
    catalogo = carica_catalogo()
    nomi = list(catalogo)
    if valore is None:
        if nomi:
            print("\nSatelliti disponibili (satellites_id.csv):")
            for k, nome in enumerate(nomi, 1):
                print(f" [{k:>2}] {nome} ({catalogo[nome]})")
            valore = input(
                "Scegli numeri, nomi o ID Horizons separati da virgole ('tutti' per tutti): "
            )
        else:
            valore = input("Inserisci nomi o ID Horizons (es. LRO, -85), separati da virgole: ")

    scelti = {}
    for elemento in valore.split(","):
        e = elemento.strip().strip("\"'")
        if not e:
            continue
        if e.lower() in ("tutti", "all"):
            scelti.update(catalogo)
        elif e.isdigit() and 1 <= int(e) <= len(nomi):  # numero del menu
            scelti[nomi[int(e) - 1]] = catalogo[nomi[int(e) - 1]]
        else:
            nome, id_horizons = risolvi_satellite(e, catalogo)
            scelti[nome] = id_horizons
    if not scelti:
        raise SystemExit("Inserisci almeno un satellite.")
    return list(scelti.items())


def main():
    adesso = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    p = argparse.ArgumentParser(
        description="Calcola quali zone della Luna vedono satelliti lunari catalogati in JPL Horizons."
    )
    p.add_argument(
        "--satellites",
        help="nomi di satellites_id.csv, 'tutti' o ID Horizons separati da virgole "
             "(es. Danuri,-85); senza questo appare il menu",
    )
    p.add_argument("--sito", help="numero del sito nel catalogo (senza questo e senza --lat/--lon appare il menu)")
    p.add_argument("--lat", type=float, help="latitudine in gradi")
    p.add_argument("--lon", type=float, help="longitudine est in gradi")
    p.add_argument("--nome", help="nome del sito personalizzato")
    p.add_argument("--inizio", default=f"{adesso:%Y-%m-%d %H:%M}", help="istante iniziale UTC (predefinito: adesso)")
    p.add_argument("--fine", help="istante finale UTC (predefinito: inizio + 30 giorni, un giorno lunare)")
    p.add_argument("--passo", default="12 h", help="passo temporale, es. '12 h', '6 h', '1 d'")
    p.add_argument("--larghezza-km", type=float, default=100.0, help="lato della zona mostrata, in km")
    p.add_argument("--raggio-km", type=float, default=150.0,
                   help="raggio del terreno considerato per l'orizzonte locale (stesso ritaglio di horizon.py)")
    p.add_argument("--punti", type=int, default=241, help="punti per lato della griglia mostrata")
    p.add_argument("--sorgente", default=USGS_LOLA_118M_URL, help="URL o file locale del DEM globale")
    p.add_argument("--no-browser", action="store_true", help="non aprire il browser alla fine")
    a = p.parse_args()

    satelliti = scegli_satelliti(a.satellites)
    nome, lat, lon = scegli_sito(a)
    lon = ((lon + 180.0) % 360.0) - 180.0
    slug = "".join(c if c.isalnum() else "_" for c in nome.lower()).strip("_")
    fine = a.fine or f"{datetime.strptime(a.inizio, '%Y-%m-%d %H:%M') + timedelta(days=30):%Y-%m-%d %H:%M}"
    raggio_m = a.raggio_km * 1000.0
    semilato_m = a.larghezza_km * 500.0
    if semilato_m >= raggio_m:
        raise SystemExit("--larghezza-km deve essere minore di 2 x --raggio-km.")

    # 1. Terreno (stesso ritaglio e stessa cache di horizon.py)
    file_tif = scarica_ritaglio(a.sorgente, slug, lat, lon, raggio_m)
    dem, transform, crs, _ = carica_ritaglio(file_tif)
    passo_m = a.larghezza_km * 1000.0 / (a.punti - 1)
    print(f"Griglia locale: passo {passo_m:.0f} m, orizzonte calcolato fino a {a.raggio_km:.0f} km dal sito...")
    coord, z_ext = griglia_locale(dem, transform, crs, lat, lon, raggio_m, passo_m)
    n_ext = (len(coord) - 1) // 2
    n_mezzo = (a.punti - 1) // 2
    i0, i1 = n_ext - n_mezzo, n_ext + n_mezzo + 1

    # 2. Common terrain data for all requested spacecraft.
    distanze = distanze_di_marcia(passo_m, math.hypot(semilato_m, semilato_m) + raggio_m)
    validi = ~np.isnan(z_ext[i0:i1, i0:i1])
    z_mostrato = z_ext[i0:i1, i0:i1]
    quota_min = float(np.nanmin(z_mostrato))
    x_km = coord[i0:i1] / 1000.0
    z_km = (z_mostrato - quota_min) / 1000.0

    sito_alt_m = float(z_ext[n_ext, n_ext])
    if not math.isfinite(sito_alt_m):
        raise SystemExit("Il DEM non contiene una quota valida al centro del sito.")

    for satellite, id_horizons in satelliti:
        print(
            f"Richiesta a JPL Horizons: {satellite} ({id_horizons}) "
            f"da {a.inizio} a {fine} ogni {a.passo}..."
        )
        try:
            ephemeris = get_lunar_ephemeris(
                id_horizons, lat, lon, sito_alt_m / 1000.0, a.inizio, fine, a.passo
            )
        except ValueError as errore:
            # Es. missione conclusa: Horizons non ha effemeridi nell'intervallo richiesto.
            testo = str(errore)
            k = testo.find("No ephemeris")
            motivo = " ".join(testo[k if k >= 0 else 0 :][:200].split())
            print(f"{satellite}: saltato, nessuna effemeride disponibile. {motivo}")
            continue

        fotogrammi, righe_csv = [], []
        for k, (istante, az, el, range_km) in enumerate(ephemeris):
            visibilita = visibilita_satellite(
                z_ext,
                passo_m,
                n_ext,
                i0,
                i1,
                az,
                el,
                range_km,
                sito_alt_m,
                distanze,
            )
            visibilita[~validi] = 0.0
            perc = 100.0 * float(np.mean(visibilita[validi])) if validi.any() else 0.0
            fotogrammi.append((istante, az, el, in_scala_colori(visibilita), perc))
            righe_csv.append(
                f"{istante:%Y-%m-%dT%H:%M}Z,{az:.4f},{el:.4f},{range_km:.3f},{perc:.2f}"
            )
            print(f"\r  {k + 1}/{len(ephemeris)} istanti", end="", flush=True)
        print()

        satellite_slug = "".join(
            c.lower() if c.isalnum() else "_" for c in satellite
        ).strip("_")
        base = f"visibilita_sat_{satellite_slug}_{slug}"
        with open(base + ".csv", "w") as f:
            f.write(
                "istante_utc,azimut_gradi,elevazione_gradi,distanza_km,"
                "percentuale_terreno_visibile\n"
            )
            f.write("\n".join(righe_csv) + "\n")
        disegna(nome, satellite, x_km, x_km, z_km, fotogrammi, base + ".html")

        perc = [fotogramma[4] for fotogramma in fotogrammi]
        print(
            f"{satellite}: terreno visibile minimo {min(perc):.1f}%, "
            f"massimo {max(perc):.1f}%, media {np.mean(perc):.1f}%"
        )
        print(f"Salvati: {base}.html e {base}.csv")
        if not a.no_browser:
            webbrowser.open("file://" + os.path.abspath(base + ".html"))


if __name__ == "__main__":
    main()
