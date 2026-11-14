"""Render Earth-visible regions on lunar terrain using JPL Horizons and LOLA."""

import argparse
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
RAGGIO_TERRA_GRADI = 0.95  # raggio apparente medio della Terra vista dalla Luna
SCALA_VISIBILITA = [
    [0.0, "#111820"],
    [1.0 / 255.0, "#111820"],
    [1.0 / 255.0, "#14532d"],
    [128.0 / 255.0, "#22c55e"],
    [1.0, "#bbf7d0"],
]


# --------------------------------------------------------------------------
# Effemeridi da JPL Horizons
# --------------------------------------------------------------------------
def get_lunar_ephemeris(target_id, lat, lon, alt_km, start_time, stop_time, step_size="1 h"):
    """
    Azimut ed elevazione di un corpo celeste visto da un punto della superficie lunare.

    target_id: '399' per la Terra
    lat, lon: coordinate selenografiche in gradi (longitudine est)
    alt_km: quota sul raggio di riferimento, in km
    start_time, stop_time: es. '2027-01-01 00:00'
    step_size: es. '1 h', '30 m', '1 d'

    Restituisce una lista di tuple (datetime UTC, azimut in gradi, elevazione in gradi).
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
        "QUANTITIES": "'4'",  # azimut ed elevazione apparenti
        "ANG_FORMAT": "DEG",
        "CSV_FORMAT": "YES",
    }
    url = HORIZONS_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as risposta:
        testo = risposta.read().decode("utf-8", errors="replace")

    # Horizons racchiude la tabella tra i marcatori $$SOE e $$EOE
    i_soe, i_eoe = testo.find("$$SOE"), testo.find("$$EOE")
    if i_soe == -1 or i_eoe == -1:
        raise ValueError(f"Tabella delle effemeridi non trovata. Risposta di Horizons:\n{testo[:1500]}")

    righe = []
    # Colonne: data, flag solare, flag lunare, azimut, elevazione, (vuota)
    for riga in testo[i_soe + 5 : i_eoe].strip().splitlines():
        campi = [c.strip() for c in riga.split(",")]
        if len(campi) < 5:
            continue
        istante = datetime.strptime(campi[0], "%Y-%b-%d %H:%M").replace(tzinfo=timezone.utc)
        righe.append((istante, float(campi[3]), float(campi[4])))
    if not righe:
        raise ValueError("Horizons ha restituito una tabella vuota.")
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
    di riferimento la direzione della Terra e' la stessa per tutti i punti.

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
    """Campioni lungo il raggio verso la Terra: fitti vicino, poi crescono del 2%."""
    d = list(np.arange(1, 41) * passo_m)
    while d[-1] < d_max:
        d.append(d[-1] + max(passo_m, 0.02 * d[-1]))
    return np.array(d)


# --------------------------------------------------------------------------
# Illuminazione
# --------------------------------------------------------------------------
def frazione_disco(el_deg, orizzonte_deg):
    """Fraction of Earth's disk above the terrain horizon (0 hidden, 1 fully visible)."""
    x = np.clip((el_deg - orizzonte_deg) / RAGGIO_TERRA_GRADI, -1.0, 1.0)
    return 1.0 - (np.arccos(x) - x * np.sqrt(1.0 - x**2)) / np.pi


def visibilita_terra(z_ext, passo_m, n_ext, i0, i1, az_deg, el_deg, distanze):
    """
    Fraction of Earth's disk visible from each displayed terrain point.

    The terrain is sampled along the azimuth toward Earth; its highest apparent
    angle sets the local horizon. Unlike sunlight, visibility does not depend
    on the terrain's surface normal.
    """
    z0 = z_ext[i0:i1, i0:i1]
    coord = (np.arange(i0, i1) - n_ext) * passo_m
    e0, n0 = np.meshgrid(coord, coord)

    # Terrain horizon in the direction of Earth.
    tan_max = np.full(z0.shape, -np.inf)
    tan_limite = math.tan(math.radians(el_deg - RAGGIO_TERRA_GRADI))
    z_top = float(np.nanmax(z_ext))
    z0_min = float(np.nanmin(z0))
    s_e, s_n = math.sin(math.radians(az_deg)), math.cos(math.radians(az_deg))
    for d in distanze:
        # Once the terrain can no longer reach Earth's lower limb, farther
        # samples cannot change the visible fraction.
        if el_deg - RAGGIO_TERRA_GRADI > 0 and (z_top - z0_min) / d < tan_limite:
            break
        zt = campiona_griglia(z_ext, passo_m, n_ext, e0 + d * s_e, n0 + d * s_n)
        np.maximum(tan_max, (zt - z0) / d, out=tan_max)
    orizzonte = np.degrees(np.arctan(tan_max))
    return frazione_disco(el_deg, orizzonte)


def in_scala_colori(visibilita):
    """Map Earth visibility fraction to the 0..255 green terrain colorscale."""
    v = np.where(visibilita > 1e-3, 1.0 + 254.0 * np.clip(visibilita, 0.0, 1.0), 0.0)
    return np.where(np.isnan(v), 0, v).astype(np.uint8)


# --------------------------------------------------------------------------
# Rendering 3D animato
# --------------------------------------------------------------------------
def disegna(nome, lat, lon, x_km, y_km, z_km, quota_min_m, fotogrammi, file_html):
    """fotogrammi: lista di (istante, azimut, elevazione, colori uint8, % Terra visibile)."""
    rilievo_km = float(np.nanmax(z_km))
    centro = len(x_km) // 2
    z_centro = float(np.nan_to_num(z_km[centro, centro]))
    lato_km = float(x_km[-1] - x_km[0])
    lunghezza_freccia = 0.35 * lato_km

    def freccia_terra(az, el):
        s = direzione_enu(az, el) * lunghezza_freccia
        colore = "#4ade80" if el > -RAGGIO_TERRA_GRADI else "#6c757d"
        return go.Scatter3d(
            x=[0.0, s[0]], y=[0.0, s[1]], z=[z_centro, z_centro + s[2]],
            mode="lines+markers+text",
            line=dict(color=colore, width=6),
            marker=dict(size=[3, 9], color=colore),
            text=["", "Terra"], textposition="top center", textfont=dict(color=colore, size=13),
            hovertemplate=f"Terra<br>Azimut {az:.1f}°<br>Elevazione {el:.2f}°<extra></extra>",
            showlegend=False,
        )

    def titolo(istante, az, el, perc):
        return (
            f"<b>Visibilità della Terra - {nome}</b><br><span style='font-size:12px;color:#aaa'>"
            f"{istante:%Y-%m-%d %H:%M} UTC | Terra: azimut {az:.1f}°, elevazione {el:+.2f}° | "
            f"superficie con Terra visibile {perc:.1f}%</span>"
        )

    def superficie(colori):
        return go.Surface(
            x=x_km, y=y_km, z=z_km,
            surfacecolor=colori, cmin=0, cmax=255, colorscale=SCALA_VISIBILITA,
            colorbar=dict(
                title=dict(text="Terra visibile", font=dict(size=12)), thickness=14, len=0.6,
                tickvals=[0, 64, 128, 192, 255],
                ticktext=["nascosta", "minima", "parziale", "quasi tutta", "tutta"],
            ),
            # Visibility is calculated from terrain horizons; Plotly adds relief shading.
            lighting=dict(ambient=0.8, diffuse=0.25, specular=0.0, roughness=1.0, fresnel=0.0),
            contours=dict(x=dict(highlight=False), y=dict(highlight=False), z=dict(highlight=False)),
            hovertemplate=(
                "Ovest-Est: %{x:.1f} km<br>Sud-Nord: %{y:.1f} km<br>"
                "Quota relativa: %{z:.2f} km<br>Luce: %{surfacecolor}/255<extra></extra>"
            ),
        )

    t0, az0, el0, col0, perc0 = fotogrammi[0]
    fig = go.Figure(data=[superficie(col0), freccia_terra(az0, el0)])
    fig.frames = [
        go.Frame(
            name=str(k),
            data=[go.Surface(surfacecolor=col), freccia_terra(az, el)],
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
        return a.nome or "Sito personalizzato", a.lat, a.lon
    scelta = a.sito
    if scelta is None:
        print("\n=== VISIBILITA' DELLA TERRA DALLA LUNA ===")
        for chiave, (nome, lat, lon) in SITI.items():
            print(f" [{chiave:>2}] {nome} ({lat}°, {lon}°)")
        print(" [ C] Latitudine e longitudine personalizzate")
        scelta = input("\nScegli un sito (predefinito = 1, Tycho): ").strip().upper() or "1"
    if scelta == "C":
        nome = input("Nome del sito: ").strip() or "Sito personalizzato"
        lat = float(input("Latitudine (-90 a 90): ").strip())
        lon = float(input("Longitudine est (-180 a 360): ").strip())
        return nome, lat, lon
    if scelta not in SITI:
        print("Scelta non valida, uso Tycho.")
        scelta = "1"
    return SITI[scelta]


def main():
    adesso = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    p = argparse.ArgumentParser(description="Zone della Luna da cui la Terra e' visibile, usando LOLA e JPL Horizons.")
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

    # 2. Posizione della Terra vista dal sito sulla Luna
    print(f"Richiesta a JPL Horizons: Terra da {a.inizio} a {fine} ogni {a.passo}...")
    terra = get_lunar_ephemeris("399", lat, lon, 0, a.inizio, fine, a.passo)

    # 3. Visibilita' della Terra per ogni istante
    distanze = distanze_di_marcia(passo_m, math.hypot(semilato_m, semilato_m) + raggio_m)
    validi = ~np.isnan(z_ext[i0:i1, i0:i1])
    fotogrammi, righe_csv = [], []
    for k, (istante, az, el) in enumerate(terra):
        visibilita = visibilita_terra(z_ext, passo_m, n_ext, i0, i1, az, el, distanze)
        visibilita[~validi] = 0.0
        perc = 100.0 * float(np.mean(visibilita[validi] > 1e-3)) if validi.any() else 0.0
        fotogrammi.append((istante, az, el, in_scala_colori(visibilita), perc))
        righe_csv.append(f"{istante:%Y-%m-%dT%H:%M}Z,{az:.4f},{el:.4f},{perc:.2f}")
        print(f"\r  {k + 1}/{len(terra)} istanti", end="", flush=True)
    print()

    # 4. Risultati
    z_mostrato = z_ext[i0:i1, i0:i1]
    quota_min = float(np.nanmin(z_mostrato))
    x_km = coord[i0:i1] / 1000.0
    base = f"visibilita_terra_{slug}"
    with open(base + ".csv", "w") as f:
        f.write("istante_utc,azimut_terra_gradi,elevazione_terra_gradi,percentuale_terra_visibile\n")
        f.write("\n".join(righe_csv) + "\n")
    disegna(nome, lat, lon, x_km, x_km, (z_mostrato - quota_min) / 1000.0, quota_min, fotogrammi, base + ".html")

    perc = [f[4] for f in fotogrammi]
    print(f"Superficie con Terra visibile: minimo {min(perc):.1f}%, massimo {max(perc):.1f}%, media {np.mean(perc):.1f}%")
    print(f"Salvati: {base}.html e {base}.csv")
    if not a.no_browser:
        webbrowser.open("file://" + os.path.abspath(base + ".html"))


if __name__ == "__main__":
    main()
