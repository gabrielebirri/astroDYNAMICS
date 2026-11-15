"""
Linea dell'orizzonte lunare a 360 gradi, con i dati del DEM LOLA globale USGS (118 m).

Uso interattivo (menu dei siti, come nell'altro script):
    python orizzonte_usgs.py

Uso diretto:
    python orizzonte_usgs.py --sito 11
    python orizzonte_usgs.py --lat -89.45 --lon 222.69 --nome "Cresta di Shackleton"

Dipendenze:
    pip install numpy rasterio pyproj plotly

Il ritaglio di DEM viene scaricato dal server USGS solo la prima volta e poi
riusato dalla cartella lunar_dems/.
"""

import argparse
import math
import os
import webbrowser

import numpy as np
import plotly.graph_objects as go
import rasterio
from affine import Affine
from plotly.colors import sample_colorscale
from pyproj import Transformer
from rasterio.windows import Window

# Lettura via rete di una sola finestra di pixel (stessa configurazione dell'altro script)
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "50000000"

USGS_LOLA_118M_URL = (
    "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
)

R_LUNA = 1_737_400.0  # raggio di riferimento della Luna, in metri
LONLAT = "+proj=longlat +R=1737400 +no_defs"
N_PIANI = 24  # numero di piani di profondita' disegnati nel panorama

# Catalogo: (nome, latitudine, longitudine)
SITI = {
    "1": ("Tycho Crater", -43.31, -11.36),
    "2": ("Copernicus Crater", 9.62, -20.08),
    "3": ("Aristarchus & Vallis Schroteri", 24.5, -49.0),
    "4": ("Vallis Alpes", 49.2, 3.2),
    "5": ("Apollo 15 - Mons Hadley", 26.13, 3.63),
    "6": ("Apollo 17 - Taurus-Littrow", 20.19, 30.77),
    "7": ("Plato Crater", 51.6, -9.3),
    "8": ("Montes Apenninus", 19.5, -1.5),
    "9": ("Rupes Recta", -21.8, -7.8),
    "10": ("Tsiolkovskiy Crater", -20.4, 129.1),
    "11": ("Polo sud - Cresta di Shackleton", -89.45, 222.69),
    "12": ("Polo sud - Massiccio Malapert", -85.99, -2.93),
    "13": ("Apollo 11 - Tranquility Base", 0.67416, 23.47314),
}


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
def base_locale(lat_deg, lon_deg):
    """Versori 'alto', 'est' e 'nord' nel punto dato (coordinate cartesiane lunari)."""
    phi, lam = math.radians(lat_deg), math.radians(lon_deg)
    alto = np.array([math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)])
    est = np.array([-math.sin(lam), math.cos(lam), 0.0])
    nord = np.array([-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)])
    return alto, est, nord


def punti_lungo_i_raggi(lat, lon, az_deg, d_m):
    """Latitudine e longitudine dei punti a distanza d_m lungo ogni azimut (matrice az x d)."""
    alto, est, nord = base_locale(lat, lon)
    a = np.radians(np.asarray(az_deg))[:, None, None]
    delta = np.asarray(d_m) / R_LUNA
    orizz = np.cos(a) * nord + np.sin(a) * est
    u = np.cos(delta)[None, :, None] * alto + np.sin(delta)[None, :, None] * orizz
    lat_p = np.degrees(np.arcsin(np.clip(u[..., 2], -1.0, 1.0)))
    lon_p = np.degrees(np.arctan2(u[..., 1], u[..., 0]))
    return u, lat_p, lon_p


def campiona(dem, transform, x, y):
    """Quota del DEM nei punti (x, y) con interpolazione bilineare. Fuori dal DEM -> NaN."""
    col, row = (~transform) * (x, y)
    col = np.asarray(col) - 0.5
    row = np.asarray(row) - 0.5
    righe, colonne = dem.shape
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    valido = (c0 >= 0) & (c0 < colonne - 1) & (r0 >= 0) & (r0 < righe - 1)
    c0 = np.clip(c0, 0, colonne - 2)
    r0 = np.clip(r0, 0, righe - 2)
    fx = col - c0
    fy = row - r0
    q = (
        dem[r0, c0] * (1 - fx) * (1 - fy)
        + dem[r0, c0 + 1] * fx * (1 - fy)
        + dem[r0 + 1, c0] * (1 - fx) * fy
        + dem[r0 + 1, c0 + 1] * fx * fy
    )
    return np.where(valido, q, np.nan)


# --------------------------------------------------------------------------
# Scarico del ritaglio dal DEM globale
# --------------------------------------------------------------------------
def scarica_ritaglio(sorgente, slug, lat, lon, raggio_m):
    """Scarica dal DEM globale la zona entro 'raggio_m' dal sito e la salva in lunar_dems/."""
    os.makedirs("lunar_dems", exist_ok=True)
    file_out = os.path.join("lunar_dems", f"{slug}_orizzonte_{int(raggio_m / 1000)}km.tif")
    if os.path.exists(file_out):
        print(f"Trovato ritaglio gia' scaricato: {file_out}")
        return file_out

    print("Connessione al DEM globale...")
    with rasterio.open(sorgente) as src:
        tr = Transformer.from_crs(LONLAT, src.crs, always_xy=True)
        # Contorno della zona da leggere: i punti alla distanza massima in tutte le direzioni
        az = np.arange(0.0, 360.0, 1.0)
        d = np.linspace(0.0, raggio_m * 1.03, 40)
        _, lat_p, lon_p = punti_lungo_i_raggi(lat, lon, az, d)
        x, y = tr.transform(lon_p, lat_p)
        col, row = (~src.transform) * (np.asarray(x), np.asarray(y))
        r_min, r_max = int(np.floor(row.min())) - 2, int(np.ceil(row.max())) + 2
        c_min, c_max = int(np.floor(col.min())) - 2, int(np.ceil(col.max())) + 2

        # Se il cerchio contiene un polo o attraversa il meridiano 180, serve tutta la larghezza
        raggio_gradi = math.degrees(raggio_m * 1.03 / R_LUNA)
        contiene_polo = abs(lat) + raggio_gradi >= 90.0
        if contiene_polo or (c_max - c_min) > src.width // 2:
            c_min, c_max = 0, src.width
            if contiene_polo and lat < 0:
                r_max = src.height
            elif contiene_polo:
                r_min = 0
        r_min, r_max = max(r_min, 0), min(r_max, src.height)
        c_min, c_max = max(c_min, 0), min(c_max, src.width)

        # Vicino ai poli i pixel sono molto stretti in direzione est-ovest:
        # si mediano k colonne alla volta per non tenere in memoria dati ridondanti.
        _, lat_bordo = Transformer.from_crs(src.crs, LONLAT, always_xy=True).transform(
            [0.0, 0.0], [src.transform.f + r_min * src.transform.e, src.transform.f + r_max * src.transform.e]
        )
        cos_min = max(math.cos(math.radians(min(abs(v) for v in lat_bordo))), 1e-3)
        k = max(1, 2 ** int(math.floor(math.log2(1.0 / cos_min))))
        larghezza = ((c_max - c_min) // k) * k
        altezza = r_max - r_min
        print(f"Scarico {larghezza} x {altezza} pixel (circa {larghezza * altezza * 2 / 1e6:.0f} MB dal server)...")

        scala = src.scales[0] if src.scales and src.scales[0] != 1.0 else (0.5 if src.dtypes[0] == "int16" else 1.0)
        offset = src.offsets[0] if src.offsets else 0.0
        dem = np.empty((altezza, larghezza // k), dtype=np.float32)
        passo_righe = 256
        for r in range(0, altezza, passo_righe):
            n = min(passo_righe, altezza - r)
            striscia = src.read(1, window=Window(c_min, r_min + r, larghezza, n)).astype(np.float32)
            if src.nodata is not None:
                striscia[striscia == src.nodata] = np.nan
            striscia = striscia * scala + offset
            dem[r : r + n] = striscia.reshape(n, larghezza // k, k).mean(axis=2)
            print(f"\r  {min(r + n, altezza)}/{altezza} righe", end="", flush=True)
        print()

        transform = src.window_transform(Window(c_min, r_min, larghezza, altezza)) * Affine.scale(k, 1)
        profilo = dict(driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1,
                       dtype="float32", crs=src.crs, transform=transform, compress="lzw", nodata=np.nan)

    # Alcuni prodotti sono in km o contengono il raggio completo: porta tutto a quote in metri
    mediana, massimo = float(np.nanmedian(dem)), float(np.nanmax(np.abs(dem)))
    if mediana > 1e6:
        dem -= R_LUNA
    elif massimo < 100:
        dem *= 1000.0

    with rasterio.open(file_out, "w", **profilo) as dst:
        dst.write(dem, 1)
    print(f"Salvato: {file_out} ({os.path.getsize(file_out) / 1e6:.1f} MB)")
    return file_out


def carica_ritaglio(file_tif):
    """Carica il ritaglio locale: quote in metri, trasformazione, proiezione, passo in metri."""
    with rasterio.open(file_tif) as src:
        dem = src.read(1).astype(np.float64)
        transform, crs = src.transform, src.crs
        passo = abs(src.transform.e)
    # Se il ritaglio copre tutti i 360 gradi, ripete la prima colonna in fondo
    # per poter interpolare anche a cavallo del meridiano 180.
    tr = Transformer.from_crs(LONLAT, crs, always_xy=True)
    giro = abs(tr.transform(180.0, 0.0)[0] - tr.transform(-180.0, 0.0)[0])
    if abs(dem.shape[1] * abs(transform.a) - giro) < abs(transform.a):
        dem = np.concatenate([dem, dem[:, :1]], axis=1)
    return dem, transform, crs, passo


# --------------------------------------------------------------------------
# Calcolo dell'orizzonte
# --------------------------------------------------------------------------
def calcola_orizzonte(dem, transform, crs, lat, lon, h_oss, raggio_m, passo_az, passo_d):
    """
    Per ogni azimut (0 = nord, 90 = est) trova l'angolo di elevazione massimo del
    terreno lungo il raggio, tenendo conto della curvatura lunare.
    """
    alto, _, _ = base_locale(lat, lon)
    tr = Transformer.from_crs(LONLAT, crs, always_xy=True)

    x0, y0 = tr.transform(lon, lat)
    quota0 = float(campiona(dem, transform, np.array([x0]), np.array([y0]))[0])
    if math.isnan(quota0):
        raise SystemExit("Nel punto scelto il DEM non contiene dati.")
    occhio = (R_LUNA + quota0 + h_oss) * alto

    az = np.arange(0.0, 360.0, passo_az)
    d = np.arange(passo_d, raggio_m + passo_d, passo_d)
    # Distanze limite dei piani, piu' fitte vicino all'osservatore (spaziatura logaritmica)
    limiti = list(np.geomspace(min(1000.0, raggio_m / 10.0), raggio_m, N_PIANI))
    indici = [min(int(np.searchsorted(d, lim, side="right")) - 1, len(d) - 1) for lim in limiti]

    elev_piani = np.full((len(limiti), len(az)), -90.0)
    dist_km = np.zeros(len(az))
    blocco = 60
    for i in range(0, len(az), blocco):
        u, lat_p, lon_p = punti_lungo_i_raggi(lat, lon, az[i : i + blocco], d)
        x, y = tr.transform(lon_p, lat_p)
        quota = campiona(dem, transform, x, y)
        v = (R_LUNA + quota)[..., None] * u - occhio
        elev = np.degrees(np.arcsin(np.clip((v @ alto) / np.linalg.norm(v, axis=-1), -1.0, 1.0)))
        elev = np.where(np.isnan(elev), -90.0, elev)
        cumulato = np.maximum.accumulate(elev, axis=1)
        for j, idx in enumerate(indici):
            elev_piani[j, i : i + blocco] = cumulato[:, idx]
        dist_km[i : i + blocco] = d[np.argmax(elev, axis=1)] / 1000.0

    piani = [(lim / 1000.0, elev_piani[j]) for j, lim in enumerate(limiti)]
    return az, piani, dist_km, quota0


def terra_media(lat, lon):
    """Azimut ed elevazione della posizione media della Terra (punto sub-terrestre a 0, 0)."""
    alto, est, nord = base_locale(lat, lon)
    verso_terra = np.array([1.0, 0.0, 0.0])
    az = math.degrees(math.atan2(verso_terra @ est, verso_terra @ nord)) % 360.0
    el = math.degrees(math.asin(verso_terra @ alto))
    return az, el


# --------------------------------------------------------------------------
# Panorama interattivo
# --------------------------------------------------------------------------
def disegna(nome, az, piani, dist_km, lat, lon, h_oss, quota0, file_html, scala="Magma"):
    orizzonte = piani[-1][1]
    y_min = math.floor(min(float(orizzonte.min()), -1.0) - 2.0)
    y_max = math.ceil(max(float(orizzonte.max()), 1.0) + 4.0)

    # Tre copie affiancate (-360, 0, +360) per ruotare senza interruzioni
    x3 = np.concatenate([az - 360.0, az, az + 360.0])
    tre = lambda v: np.concatenate([v, v, v])

    # Colore di ogni piano in base alla distanza (scala logaritmica):
    # chiaro = rilievi vicini, scuro = rilievi lontani.
    lim_km = np.array([p[0] for p in piani])
    t = (np.log(lim_km) - np.log(lim_km[0])) / (np.log(lim_km[-1]) - np.log(lim_km[0]))
    colori = sample_colorscale(scala, [float(0.90 - 0.68 * v) for v in t])

    fig = go.Figure()
    for k in range(len(piani) - 1, -1, -1):  # dal piano piu' lontano al piu' vicino
        fig.add_trace(go.Scatter(x=x3, y=np.full(x3.shape, y_min), mode="lines",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=x3, y=tre(piani[k][1]), mode="lines", fill="tonexty", fillcolor=colori[k],
            line=dict(color="rgba(0,0,0,0.45)", width=0.7),  # bordo scuro che stacca le creste
            hoverinfo="skip", showlegend=False,
        ))
    # Linea dell'orizzonte vera e propria, con i dettagli al passaggio del mouse
    fig.add_trace(go.Scatter(
        x=x3, y=tre(orizzonte), mode="lines", line=dict(color="#ffffff", width=1.6),
        name="linea dell'orizzonte", customdata=np.stack([tre(az), tre(dist_km)], axis=-1),
        hovertemplate=("Azimut %{customdata[0]:.1f}°<br>Orizzonte a %{y:.2f}°<br>"
                       "Rilievo a %{customdata[1]:.1f} km<extra></extra>"),
    ))
    # Barra dei colori: distanza dei rilievi
    tacche = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 300) if lim_km[0] <= v <= lim_km[-1]]
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(
            color=[0.0], cmin=0.0, cmax=1.0, showscale=True,
            colorscale=[[float(v), c] for v, c in zip(t, colori)],
            colorbar=dict(
                title=dict(text="Distanza<br>del rilievo", font=dict(size=12)), thickness=14, len=0.75,
                tickvals=[float((math.log(v) - math.log(lim_km[0])) / (math.log(lim_km[-1]) - math.log(lim_km[0]))) for v in tacche],
                ticktext=[f"{v} km" for v in tacche],
            ),
        ),
    ))
    fig.add_hline(y=0, line=dict(color="#f2c14e", width=1, dash="dot"))

    # La Terra viene disegnata solo se e' vicina all'orizzonte (siti polari o al bordo del disco)
    az_t, el_t = terra_media(lat, lon)
    terra_bassa = abs(el_t) < 15.0
    if terra_bassa:
        for s in (-360.0, 0.0, 360.0):
            fig.add_shape(type="rect", x0=az_t + s - 8, x1=az_t + s + 8, y0=el_t - 6.7, y1=el_t + 6.7,
                          line=dict(color="#4ea8de", width=1, dash="dot"), fillcolor="rgba(78,168,222,0.08)")
        fig.add_trace(go.Scatter(
            x=[az_t - 360.0, az_t, az_t + 360.0], y=[el_t] * 3, mode="markers+text",
            marker=dict(size=14, color="#4ea8de", line=dict(color="white", width=1)),
            text=["Terra (media)"] * 3, textposition="top center", textfont=dict(color="#4ea8de"),
            name="Terra, posizione media",
            hovertemplate=f"Terra, posizione media<br>Azimut {az_t:.1f}°<br>Elevazione {el_t:.2f}°<extra></extra>",
        ))
        y_max = max(y_max, math.ceil(el_t + 9.0))
        y_min = min(y_min, math.floor(el_t - 9.0))
        nota_terra = ""
    elif el_t > 0:
        nota_terra = f" | Terra alta nel cielo (circa {el_t:.0f}°, azimut {az_t:.0f}°)"
    else:
        nota_terra = " | Terra mai visibile (lato nascosto)"

    nomi = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    tick = np.arange(-360, 721, 15)
    etichette = [f"{nomi[int(t % 360) // 45]} {int(t % 360)}°" if t % 45 == 0 else f"{int(t % 360)}°" for t in tick]
    viste = [("Nord", 0), ("Est", 90), ("Sud", 180), ("Ovest", 270)] + ([("Terra", az_t)] if terra_bassa else [])
    centro = az_t if terra_bassa else 0.0

    fig.update_layout(
        title=dict(text=(f"<b>Orizzonte a 360° - {nome}</b><br><span style='font-size:12px;color:#aaa'>"
                         f"Lat {lat:.3f}°, Lon {lon:.3f}° | quota del terreno {quota0:.0f} m | "
                         f"osservatore a {h_oss:g} m dal suolo{nota_terra} | trascina per ruotare</span>"),
                   x=0.02, y=0.96),
        template="plotly_dark", paper_bgcolor="#05060a", plot_bgcolor="#05060a",
        dragmode="pan", hovermode="closest", margin=dict(l=60, r=90, t=110, b=50),
        xaxis=dict(range=[centro - 60, centro + 60], tickvals=tick, ticktext=etichette,
                   showgrid=True, gridcolor="#1c1f26", zeroline=False, title="Direzione di osservazione"),
        yaxis=dict(range=[y_min, y_max], fixedrange=True, gridcolor="#1c1f26", zeroline=False,
                   title="Elevazione sull'orizzontale (°)"),
        legend=dict(orientation="h", y=-0.15),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.02, y=1.08, xanchor="left", showactive=False,
            bgcolor="#222", font=dict(color="white", size=11),
            buttons=[dict(label=n, method="relayout", args=[{"xaxis.range": [c - 60, c + 60]}]) for n, c in viste],
        )],
    )
    fig.write_html(file_html, config={"scrollZoom": False, "displaylogo": False})
    return fig


# --------------------------------------------------------------------------
def scegli_sito(a):
    """Restituisce (nome, lat, lon) dagli argomenti o dal menu interattivo."""
    if a.lat is not None and a.lon is not None:
        return a.nome or "Sito personalizzato", a.lat, a.lon
    scelta = a.sito
    if scelta is None:
        print("\n=== ORIZZONTE LUNARE A 360 GRADI ===")
        for chiave, (nome, lat, lon) in SITI.items():
            print(f" [{chiave:>2}] {nome} ({lat}°, {lon}°)")
        print(" [ C] Latitudine e longitudine personalizzate")
        scelta = input("\nScegli un sito (predefinito = 11, cresta di Shackleton): ").strip().upper() or "11"
    if scelta == "C":
        nome = input("Nome del sito: ").strip() or "Sito personalizzato"
        lat = float(input("Latitudine (-90 a 90): ").strip())
        lon = float(input("Longitudine est (-180 a 360): ").strip())
        return nome, lat, lon
    if scelta not in SITI:
        print("Scelta non valida, uso la cresta di Shackleton.")
        scelta = "11"
    return SITI[scelta]


def main():
    p = argparse.ArgumentParser(description="Panorama a 360 gradi dell'orizzonte lunare dal DEM LOLA globale USGS.")
    p.add_argument("--sito", help="numero del sito nel catalogo (senza questo e senza --lat/--lon appare il menu)")
    p.add_argument("--lat", type=float, help="latitudine in gradi")
    p.add_argument("--lon", type=float, help="longitudine est in gradi")
    p.add_argument("--nome", help="nome del sito personalizzato")
    p.add_argument("--altezza", type=float, default=2.0, help="altezza dell'osservatore sul suolo, in metri")
    p.add_argument("--raggio-km", type=float, default=150.0, help="distanza massima del terreno considerato")
    p.add_argument("--passo-az", type=float, default=0.5, help="passo in azimut, in gradi")
    p.add_argument("--colori", default="Magma",
                   help="scala di colori Plotly per la profondita' (es. Viridis, Cividis, Ice, Greys_r); "
                        "aggiungere _r al nome per invertirla")
    p.add_argument("--sorgente", default=USGS_LOLA_118M_URL, help="URL o file locale del DEM globale")
    p.add_argument("--no-browser", action="store_true", help="non aprire il browser alla fine")
    a = p.parse_args()

    nome, lat, lon = scegli_sito(a)
    lon = ((lon + 180.0) % 360.0) - 180.0  # porta la longitudine tra -180 e 180
    slug = "".join(c if c.isalnum() else "_" for c in nome.lower()).strip("_")
    raggio_m = a.raggio_km * 1000.0

    file_tif = scarica_ritaglio(a.sorgente, slug, lat, lon, raggio_m)
    dem, transform, crs, passo = carica_ritaglio(file_tif)
    az, piani, dist_km, quota0 = calcola_orizzonte(
        dem, transform, crs, lat, lon, a.altezza, raggio_m, a.passo_az, passo_d=passo
    )
    orizzonte = piani[-1][1]

    base = f"orizzonte_{slug}"
    np.savetxt(base + ".csv", np.column_stack([az, orizzonte, dist_km]), delimiter=",",
               header="azimut_gradi,elevazione_orizzonte_gradi,distanza_rilievo_km", comments="", fmt="%.4f")
    disegna(nome, az, piani, dist_km, lat, lon, a.altezza, quota0, base + ".html", scala=a.colori)

    i_max, i_min = int(np.argmax(orizzonte)), int(np.argmin(orizzonte))
    print(f"Quota del terreno nel sito: {quota0:.0f} m")
    print(f"Orizzonte piu' alto: {orizzonte[i_max]:+.2f} gradi verso azimut {az[i_max]:.1f} (rilievo a {dist_km[i_max]:.1f} km)")
    print(f"Orizzonte piu' basso: {orizzonte[i_min]:+.2f} gradi verso azimut {az[i_min]:.1f}")
    print(f"Salvati: {base}.html e {base}.csv")
    if not a.no_browser:
        webbrowser.open(base + ".html")


if __name__ == "__main__":
    main()
