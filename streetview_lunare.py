"""
"Street view" lunare: genera una pagina HTML in cui ci si guarda intorno in prima
persona dal sito scelto, ci si sposta sul terreno (tastiera o clic sulla destinazione),
si passa alla vista in terza persona con un astronauta o un rover.

Uso (horizon.py ed ephemeris.py devono stare nella stessa cartella):
    python streetview_lunare.py
    python streetview_lunare.py --sito 11 --istante "2027-01-10 00:00"
    python streetview_lunare.py --sito 11 --inizio "2026-11-18 00:00" --fine "2026-12-02 00:00" --passo "6 h"

La posizione del Sole e della Terra viene chiesta a JPL Horizons con la funzione
get_lunar_ephemeris di ephemeris.py (ore UTC, al minuto):
  - --istante: un solo istante (senza opzioni si usa l'ora attuale);
  - --inizio, --fine, --passo: un intervallo; nella pagina compare un cursore per
    scegliere l'istante. Il Sole non si muove da solo: resta fermo finche' non si
    sposta il cursore.

Senza connessione si puo' usare --effemeridi skyfield (effemeridi JPL DE421 in locale,
richiede "pip install skyfield" e scarica circa 19 MB la prima volta), oppure passare
--sole-az e --sole-el gia' calcolati.

Dipendenze:
    pip install numpy rasterio pyproj plotly

Il risultato e' un unico file streetview_<sito>.html che contiene anche i dati del
terreno: si apre con un doppio clic e si puo' inserire in un sito con un <iframe>.
"""

import argparse
import base64
import datetime as dt
import json
import math
import os
import re
import webbrowser

import numpy as np
from pyproj import Transformer

try:
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
except ImportError:  # nome originale dello stesso modulo
    from orizzonte_usgs import (
        LONLAT,
        R_LUNA,
        SITI,
        USGS_LOLA_118M_URL,
        base_locale,
        campiona,
        carica_ritaglio,
        scarica_ritaglio,
    )

CARTELLA_EFFEMERIDI = "effemeridi"

MEZZO_LATO_VICINO_M = 15_000.0  # zona a pieno dettaglio attorno all'osservatore (circa)
FATTORE_LONTANO = 5  # il terreno lontano usa una griglia 5 volte piu' rada
SENZA_DATO = -32768


def griglia_locale(dem, transform, crs, lat, lon, mezzo_lato_m, passo_m):
    """
    Ricampiona il DEM su una griglia quadrata centrata sul sito, con assi est e nord
    in metri misurati lungo la superficie. Restituisce le quote in metri (NaN se mancano).
    """
    alto, est, nord = base_locale(lat, lon)
    n = 2 * int(round(mezzo_lato_m / passo_m)) + 1  # dispari: il sito e' il punto centrale
    coord = (np.arange(n) - (n - 1) / 2) * passo_m
    e, nn = np.meshgrid(coord, coord)  # riga = nord crescente, colonna = est crescente
    d = np.hypot(e, nn)
    az = np.arctan2(e, nn)
    delta = d / R_LUNA
    orizz = np.cos(az)[..., None] * nord + np.sin(az)[..., None] * est
    u = np.cos(delta)[..., None] * alto + np.sin(delta)[..., None] * orizz
    lat_p = np.degrees(np.arcsin(np.clip(u[..., 2], -1.0, 1.0)))
    lon_p = np.degrees(np.arctan2(u[..., 1], u[..., 0]))
    tr = Transformer.from_crs(LONLAT, crs, always_xy=True)
    x, y = tr.transform(lon_p, lat_p)
    return campiona(dem, transform, x, y), n


def sole_terra_da_horizons(lat, lon, quota_m, inizio, fine, passo):
    """
    Azimut ed elevazione del Sole e della Terra visti dal sito, chiesti a JPL Horizons
    con la funzione get_lunar_ephemeris di ephemeris.py per tutto l'intervallo.
    Restituisce una lista di (istante, (az_sole, el_sole), (az_terra, el_terra)).
    """
    from ephemeris import get_lunar_ephemeris

    formato = "%Y-%m-%d %H:%M"
    tabelle = [
        get_lunar_ephemeris(corpo, lat, lon % 360.0, quota_m / 1000.0, f"{inizio:{formato}}", f"{fine:{formato}}", passo)
        for corpo in ("10", "399")  # 10 = Sole, 399 = Terra
    ]
    return [(t, (az_s % 360.0, el_s), (az_t % 360.0, el_t))
            for (t, az_s, el_s), (_, az_t, el_t) in zip(*tabelle)]


def sole_terra_da_skyfield(lat, lon, istanti):
    """Come sopra, ma calcolato in locale dalle effemeridi JPL DE421 (senza connessione)."""
    return [(t, azimut_elevazione(lat, lon, sole), azimut_elevazione(lat, lon, terra))
            for t, (sole, terra) in zip(istanti, direzioni_sole_terra(istanti))]


def leggi_istante(testo):
    """Da testo ("2027-01-10 00:00" o formato ISO) a data e ora UTC, al minuto."""
    t = dt.datetime.fromisoformat(testo.strip().replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)  # senza fuso orario si intende UTC
    return t.astimezone(dt.timezone.utc).replace(second=0, microsecond=0)


def leggi_passo(testo):
    """Da "6 h", "30 m" o "1 d" a un intervallo di tempo."""
    m = re.fullmatch(r"\s*(\d+)\s*([mhd])\w*\s*", testo.lower())
    if not m or int(m.group(1)) == 0:
        raise SystemExit(f'Passo non valido: "{testo}". Esempi: "30 m", "6 h", "1 d".')
    return int(m.group(1)) * {"m": dt.timedelta(minutes=1), "h": dt.timedelta(hours=1), "d": dt.timedelta(days=1)}[m.group(2)]


def terra_media(lat, lon):
    """Azimut ed elevazione della posizione media della Terra (punto sub-terrestre a 0, 0)."""
    return azimut_elevazione(lat, lon, np.array([1.0, 0.0, 0.0]))


def scegli_sito(a):
    """Restituisce (nome, lat, lon) dagli argomenti o dal menu interattivo."""
    if a.lat is not None and a.lon is not None:
        return a.nome or "Sito personalizzato", a.lat, a.lon
    scelta = a.sito
    if scelta is None:
        print("\n=== VISTA DAL SUOLO LUNARE ===")
        for chiave, (nome, lat, lon) in SITI.items():
            print(f" [{chiave:>2}] {nome} ({lat}°, {lon}°)")
        print(" [ C] Latitudine e longitudine personalizzate")
        scelta = input("\nScegli un sito (predefinito = 1): ").strip().upper() or "1"
    if scelta == "C":
        nome = input("Nome del sito: ").strip() or "Sito personalizzato"
        lat = float(input("Latitudine (-90 a 90): ").strip())
        lon = float(input("Longitudine est (-180 a 360): ").strip())
        return nome, lat, lon
    if scelta not in SITI:
        print("Scelta non valida, uso il sito 1.")
        scelta = "1"
    return SITI[scelta][:3]


def direzioni_sole_terra(istanti):
    """
    Per ogni istante, i versori verso il Sole e verso la Terra visti dalla Luna, espressi
    nel sistema fisso con la Luna (MOON_ME, lo stesso delle coordinate del DEM).
    Usa le effemeridi JPL DE421 e l'orientazione lunare DE421 tramite Skyfield.
    """
    from skyfield.api import Loader
    from skyfield.planetarylib import PlanetaryConstants

    carica = Loader(CARTELLA_EFFEMERIDI)  # i file scaricati restano in questa cartella
    scala_tempi = carica.timescale(builtin=True)
    eph = carica("de421.bsp")
    costanti = PlanetaryConstants()
    costanti.read_text(carica("moon_080317.tf"))
    costanti.read_binary(carica("moon_pa_de421_1900-2050.bpc"))
    sistema_lunare = costanti.build_frame_named("MOON_ME_DE421")

    risultati = []
    for istante in istanti:
        t = scala_tempi.from_datetime(istante)
        rotazione = sistema_lunare.rotation_at(t)  # da assi inerziali ad assi fissi con la Luna
        coppia = []
        for corpo in ("sun", "earth"):
            v = rotazione @ eph["moon"].at(t).observe(eph[corpo]).position.km
            coppia.append(v / np.linalg.norm(v))
        risultati.append(coppia)
    return risultati


def azimut_elevazione(lat, lon, versore):
    """Azimut (0 = nord, 90 = est) ed elevazione, in gradi, di una direzione vista dal sito."""
    alto, est, nord = base_locale(lat, lon)
    az = math.degrees(math.atan2(versore @ est, versore @ nord)) % 360.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, float(versore @ alto)))))
    return az, el


def cuci_bordo(vicino, lontano, celle):
    """
    Rende il bordo della griglia vicina identico al terreno lontano: lungo il bordo le
    quote seguono i segmenti della griglia rada, cosi' le due superfici combaciano
    senza fessure ne' sovrapposizioni.
    """
    n_v, c_l, k = vicino.shape[0], (lontano.shape[0] - 1) // 2, celle // FATTORE_LONTANO
    fitto = np.arange(n_v) / FATTORE_LONTANO
    rado = np.arange(2 * k + 1)
    tratto = slice(c_l - k, c_l + k + 1)
    vicino[0, :] = np.interp(fitto, rado, lontano[c_l - k, tratto])    # bordo sud
    vicino[-1, :] = np.interp(fitto, rado, lontano[c_l + k, tratto])   # bordo nord
    vicino[:, 0] = np.interp(fitto, rado, lontano[tratto, c_l - k])    # bordo ovest
    vicino[:, -1] = np.interp(fitto, rado, lontano[tratto, c_l + k])   # bordo est
    return vicino


def impacchetta(quote, n, passo_m):
    """Quote -> interi a 16 bit in unita' di mezzo metro, codificati in base64 per l'HTML."""
    q = np.round(quote / 0.5)
    q = np.where(np.isnan(q), SENZA_DATO, np.clip(q, -32767, 32767)).astype("<i2")
    return {"n": int(n), "passo": float(passo_m), "dati": base64.b64encode(q.tobytes()).decode("ascii")}


def main():
    p = argparse.ArgumentParser(description="Genera una vista in prima persona del terreno lunare.")
    p.add_argument("--sito", help="numero del sito nel catalogo (senza questo e senza --lat/--lon appare il menu)")
    p.add_argument("--lat", type=float, help="latitudine in gradi")
    p.add_argument("--lon", type=float, help="longitudine est in gradi")
    p.add_argument("--nome", help="nome del sito personalizzato")
    p.add_argument("--istante", default="2026-11-27 02:00", help="un solo istante UTC, es. \"2027-011-27 02:00\" (predefinito: adesso)")
    p.add_argument("--inizio", help="inizio dell'intervallo UTC, es. \"2026-11-18 00:00\" (al posto di --istante)")
    p.add_argument("--fine", help="fine dell'intervallo UTC (predefinito: inizio + 14 giorni)")
    p.add_argument("--passo", default="6 h", help="passo dell'intervallo, es. \"30 m\", \"6 h\", \"1 d\" (predefinito: 6 h)")
    p.add_argument("--effemeridi", choices=["horizons", "skyfield"], default="horizons",
                   help="horizons = JPL Horizons tramite ephemeris.py (predefinito); skyfield = calcolo in locale")
    p.add_argument("--confronta", action="store_true",
                   help="con horizons, calcola anche con skyfield e stampa la differenza (controllo incrociato)")
    p.add_argument("--sole-az", type=float, help="azimut del Sole in gradi, se calcolato con un altro programma")
    p.add_argument("--sole-el", type=float, help="elevazione del Sole in gradi, se calcolata con un altro programma")
    p.add_argument("--raggio-km", type=float, default=150.0, help="distanza massima del terreno mostrato")
    p.add_argument("--sorgente", default=USGS_LOLA_118M_URL, help="URL o file locale del DEM globale")
    p.add_argument("--no-browser", action="store_true", help="non aprire il browser alla fine")
    a = p.parse_args()

    nome, lat, lon = scegli_sito(a)
    lon = ((lon + 180.0) % 360.0) - 180.0
    slug = "".join(c if c.isalnum() else "_" for c in nome.lower()).strip("_")
    raggio_m = a.raggio_km * 1000.0

    # Stesso ritaglio usato da orizzonte_usgs.py: se esiste gia' non viene riscaricato
    file_tif = scarica_ritaglio(a.sorgente, slug, lat, lon, raggio_m)
    dem, transform, crs, passo = carica_ritaglio(file_tif)

    # La zona vicina misura un numero intero di celle della griglia lontana, cosi' i bordi coincidono
    passo_l = passo * FATTORE_LONTANO
    celle = int(min(MEZZO_LATO_VICINO_M, raggio_m * 0.9) / passo_l) * FATTORE_LONTANO
    vicino, n_v = griglia_locale(dem, transform, crs, lat, lon, celle * passo, passo)
    lontano, n_l = griglia_locale(dem, transform, crs, lat, lon, raggio_m, passo_l)
    vicino = cuci_bordo(vicino, lontano, celle)
    quota0 = float(vicino[(n_v - 1) // 2, (n_v - 1) // 2])
    if math.isnan(quota0):
        raise SystemExit("Nel punto scelto il DEM non contiene dati.")

    # Posizione del Sole e della Terra: un istante solo, oppure un intervallo a scelta
    if (a.sole_az is None) != (a.sole_el is None):
        raise SystemExit("Indicare sia --sole-az sia --sole-el, oppure nessuno dei due.")
    if a.istante and a.inizio:
        raise SystemExit("Usare --istante oppure --inizio/--fine, non entrambi.")
    if a.sole_az is not None:
        cielo = [("posizione inserita a mano", (a.sole_az % 360.0, a.sole_el), terra_media(lat, lon))]
        fonte = "valori inseriti a mano, Terra in posizione media"
    else:
        if a.inizio:
            inizio = leggi_istante(a.inizio)
            fine = leggi_istante(a.fine) if a.fine else inizio + dt.timedelta(days=14)
            passo_tempo = leggi_passo(a.passo)
            if fine <= inizio:
                raise SystemExit("--fine deve essere successivo a --inizio.")
            if (fine - inizio) / passo_tempo > 2000:
                raise SystemExit("Troppi istanti (piu' di 2000): aumentare --passo o accorciare l'intervallo.")
        else:
            inizio = leggi_istante(a.istante) if a.istante else dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
            fine, passo_tempo = None, None
        if a.effemeridi == "horizons":
            try:
                # Per un istante solo si chiede a Horizons un minuto e si tiene il primo valore
                tabella = sole_terra_da_horizons(lat, lon, quota0, inizio, fine or inizio + dt.timedelta(minutes=1),
                                                 a.passo if fine else "1 m")
            except Exception as errore:
                raise SystemExit(
                    f"Richiesta a JPL Horizons non riuscita: {errore}\n"
                    "Controllare la connessione, oppure usare --effemeridi skyfield o --sole-az/--sole-el."
                )
            if not fine:
                tabella = tabella[:1]
            fonte = "JPL Horizons"
            if a.confronta:
                _, (az2, el2), _ = sole_terra_da_skyfield(lat, lon, [tabella[0][0]])[0]
                (az1, el1) = tabella[0][1]
                print(f"Controllo con Skyfield al primo istante: Sole azimut {az2:.2f}, elevazione {el2:+.2f} "
                      f"(differenze: {(az1 - az2 + 180.0) % 360.0 - 180.0:+.3f} e {el1 - el2:+.3f} gradi)")
        else:
            istanti = [inizio] if not fine else [inizio + k * passo_tempo for k in range(int((fine - inizio) / passo_tempo) + 1)]
            tabella = sole_terra_da_skyfield(lat, lon, istanti)
            fonte = "JPL DE421"
        cielo = [(f"{t:%Y-%m-%d %H:%M} UTC", sole, terra) for t, sole, terra in tabella]

    elevazioni = [sole[1] for _, sole, _ in cielo]
    print(f"Fonte delle effemeridi: {fonte}")
    if len(cielo) == 1:
        print(f"Istante: {cielo[0][0]}")
        print(f"Sole:  azimut {cielo[0][1][0]:6.2f} gradi, elevazione {cielo[0][1][1]:+6.2f} gradi")
        print(f"Terra: azimut {cielo[0][2][0]:6.2f} gradi, elevazione {cielo[0][2][1]:+6.2f} gradi")
    else:
        con_luce = sum(1 for e in elevazioni if e > 0)
        print(f"Intervallo: da {cielo[0][0]} a {cielo[-1][0]}, {len(cielo)} istanti")
        print(f"Elevazione del Sole: da {min(elevazioni):+.2f} a {max(elevazioni):+.2f} gradi; "
              f"sopra l'orizzontale in {con_luce} istanti su {len(cielo)}")
    if max(elevazioni) <= 0:
        print("Attenzione: in questo periodo il Sole resta sotto l'orizzontale, la scena sara' buia.")

    dati = {
        "nome": nome,
        "lat": lat,
        "lon": lon,
        "quota0": quota0,
        "raggio": raggio_m,
        "fonte": fonte,
        # per ogni istante: testo, [azimut, elevazione] del Sole, [azimut, elevazione] della Terra
        "cielo": [{"t": t, "s": [round(v, 4) for v in sole], "e": [round(v, 4) for v in terra]} for t, sole, terra in cielo],
        "vicino": impacchetta(vicino, n_v, passo),
        "lontano": impacchetta(lontano, n_l, passo_l),
    }

    file_html = f"streetview_{slug}.html"
    with open(file_html, "w", encoding="utf-8") as f:
        f.write(MODELLO_HTML.replace("__TITOLO__", nome).replace("__DATI__", json.dumps(dati)))

    print(f"Terreno vicino: {n_v} x {n_v} punti ogni {passo:.0f} m | terreno lontano: {n_l} x {n_l} punti ogni {passo_l:.0f} m")
    print(f"Salvato: {file_html} ({os.path.getsize(file_html) / 1e6:.1f} MB)")
    if not a.no_browser:
        webbrowser.open(file_html)


MODELLO_HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vista dal suolo - __TITOLO__</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; color: #e8eaee; overflow: hidden;
               font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
  canvas:active { cursor: grabbing; }
  .riquadro { position: absolute; background: rgba(8, 10, 14, 0.72); border: 1px solid #2a2f3a;
              border-radius: 8px; padding: 10px 12px; }
  #info { top: 12px; left: 12px; max-width: 46ch; }
  #info h1 { font-size: 16px; margin: 0 0 4px; }
  #info p { margin: 0; color: #a9afba; font-size: 12.5px; }
  #bussola { bottom: 14px; left: 50%; transform: translateX(-50%); font-variant-numeric: tabular-nums;
             white-space: nowrap; text-align: center; }
  #bussola b { font-size: 18px; }
  #comandi { top: 12px; right: 12px; width: 240px; }
  #comandi label { display: block; font-size: 12.5px; color: #a9afba; margin-top: 8px; }
  #comandi > label:first-child { margin-top: 0; }
  #comandi output { color: #e8eaee; float: right; font-variant-numeric: tabular-nums; }
  #comandi input { width: 100%; margin: 2px 0 0; }
  #comandi button { margin-top: 10px; width: 100%; padding: 6px; background: #1b1f29; color: #e8eaee;
                    border: 1px solid #3a4150; border-radius: 6px; cursor: pointer; }
  #comandi button:hover { background: #262c3a; }
  .scelta { display: flex; gap: 6px; margin-top: 3px; }
  #comandi .scelta button { margin-top: 0; flex: 1; padding: 5px 2px; font-size: 12.5px; }
  #comandi .scelta button.attivo { background: #2f5f96; border-color: #6fa3dc; }
  #comandi input:disabled { opacity: 0.35; }
  #errore { top: 50%; left: 50%; transform: translate(-50%, -50%); display: none; max-width: 40ch; }
</style>
</head>
<body>
<canvas id="vista"></canvas>
<div id="info" class="riquadro">
  <h1>__TITOLO__</h1>
  <p id="coord"></p>
  <p id="cielo"></p>
  <p>Trascina per guardarti intorno. Clic su un punto del terreno per andarci. W A S D o frecce per spostarti, Maiusc per andare più veloce. Rotella per lo zoom. V cambia vista, C cambia personaggio.</p>
</div>
<div id="comandi" class="riquadro">
  <label>Vista</label>
  <div class="scelta"><button id="vista1" type="button">Prima persona</button><button id="vista3" type="button">Terza persona</button></div>
  <label>Personaggio</label>
  <div class="scelta"><button id="persA" type="button">Astronauta</button><button id="persR" type="button">Rover</button></div>
  <label>Velocità <output id="oVel"></output><input id="vel" type="range" min="0" max="1" step="0.001"></label>
  <label id="rigaIstante">Istante <output id="oIst"></output><input id="ist" type="range" min="0" step="1" value="0"></label>
  <label>Altezza degli occhi <output id="oAlt"></output><input id="alt" type="range" min="0" max="1" step="0.001"></label>
  <button id="ritorna" type="button">Torna al punto di partenza</button>
</div>
<div id="bussola" class="riquadro"></div>
<div id="errore" class="riquadro"></div>
<script id="dati" type="application/json">__DATI__</script>
<script type="importmap">{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js" } }</script>
<script type="module">
import * as THREE from "three";

const D = JSON.parse(document.getElementById("dati").textContent);
const R = 1737400;            // raggio della Luna in metri
const SENZA_DATO = -32768;
const rad = Math.PI / 180;

// ---------- Dati del terreno ----------
for (const g of [D.vicino, D.lontano]) {
  const testo = atob(g.dati);
  const byte = new Uint8Array(testo.length);
  for (let i = 0; i < testo.length; i++) byte[i] = testo.charCodeAt(i);
  g.h = new Int16Array(byte.buffer);        // quote in unita' di mezzo metro
  g.meta = (g.n - 1) / 2 * g.passo;         // mezzo lato della griglia in metri
  delete g.dati;
}

// Quota del terreno nel punto (est, nord) di una griglia. Ogni cella e' disegnata come due
// triangoli (diagonale da sud-ovest a nord-est): qui si usa lo stesso piano del triangolo,
// cosi' la quota calcolata coincide esattamente con la superficie che si vede.
function campiona(g, e, n) {
  const x = (e + g.meta) / g.passo, y = (n + g.meta) / g.passo;
  if (x < 0 || y < 0 || x > g.n - 1 || y > g.n - 1) return null;
  const i = Math.min(Math.floor(x), g.n - 2), j = Math.min(Math.floor(y), g.n - 2);
  const fx = x - i, fy = y - j;
  const a = g.h[j * g.n + i], b = g.h[j * g.n + i + 1];
  const c = g.h[(j + 1) * g.n + i], d = g.h[(j + 1) * g.n + i + 1];
  if (a === SENZA_DATO || b === SENZA_DATO || c === SENZA_DATO || d === SENZA_DATO) return null;
  return 0.5 * (fx >= fy ? a + (b - a) * fx + (d - b) * fy : a + (d - c) * fx + (c - a) * fy);
}
function quota(e, n) {
  const v = campiona(D.vicino, e, n);
  return v !== null ? v : campiona(D.lontano, e, n);
}

// Da (est, nord, quota) a coordinate 3D della scena, con la curvatura lunare.
// Assi della scena: x = est, y = alto, z = sud (quindi il nord e' -z).
function posizione3D(e, n, h, out) {
  const d = Math.hypot(e, n);
  if (d < 1e-6) return out.set(0, h - D.quota0, 0);
  const delta = d / R, r = (R + h) * Math.sin(delta);
  return out.set(r * e / d, (R + h) * Math.cos(delta) - (R + D.quota0), -r * n / d);
}

// ---------- Scena ----------
const tela = document.getElementById("vista");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas: tela, antialias: true, logarithmicDepthBuffer: true });
} catch (err) {
  const box = document.getElementById("errore");
  box.style.display = "block";
  box.textContent = "Questo browser non riesce ad avviare la grafica 3D (WebGL).";
  throw err;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scena = new THREE.Scene();
scena.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(60, 1, 0.5, 1.2e6);

// Costruisce la superficie di una griglia. 'escludi' e' il mezzo lato (in metri) della
// zona centrale da lasciare vuota, perche' coperta dalla griglia piu' dettagliata.
function creaSuperficie(g, escludi, rumore) {
  const N = g.n, pos = new Float32Array(N * N * 3), col = new Float32Array(N * N * 3);
  const valido = new Uint8Array(N * N), v = new THREE.Vector3();
  let seme = 12345;
  const casuale = () => (seme = (seme * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const k = j * N + i, raw = g.h[k];
    if (raw === SENZA_DATO) continue;
    valido[k] = 1;
    posizione3D(i * g.passo - g.meta, j * g.passo - g.meta, raw * 0.5, v);
    pos[3 * k] = v.x; pos[3 * k + 1] = v.y; pos[3 * k + 2] = v.z;
    const c = 0.62 + rumore * (casuale() - 0.5);   // leggera variazione di tono del suolo
    col[3 * k] = c; col[3 * k + 1] = c; col[3 * k + 2] = c * 0.98;
  }
  const indici = [];
  for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
    const a = j * N + i, b = a + 1, c = a + N, d = c + 1;
    if (!(valido[a] && valido[b] && valido[c] && valido[d])) continue;
    if (escludi > 0) {
      const e0 = i * g.passo - g.meta, n0 = j * g.passo - g.meta;
      if (e0 > -escludi - 1 && e0 + g.passo < escludi + 1 && n0 > -escludi - 1 && n0 + g.passo < escludi + 1) continue;
    }
    indici.push(a, b, d, a, d, c);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indici), 1));
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }));
}
const terreni = [creaSuperficie(D.vicino, 0, 0.10), creaSuperficie(D.lontano, D.vicino.meta, 0.04)];
scena.add(...terreni);

// Luce: il Sole come unica sorgente, piu' un filo di luce diffusa per non avere nero assoluto
const luceSole = new THREE.DirectionalLight(0xffffff, 3.2);
scena.add(luceSole, luceSole.target, new THREE.AmbientLight(0xffffff, 0.05));

// Dischi del Sole e della Terra, tenuti a distanza fissa dalla camera
const DIST_CIELO = 4e5;
const discoSole = new THREE.Mesh(new THREE.SphereGeometry(DIST_CIELO * Math.tan(0.265 * rad), 24, 16),
                                 new THREE.MeshBasicMaterial({ color: 0xfff6dd }));
const discoTerra = new THREE.Mesh(new THREE.SphereGeometry(DIST_CIELO * Math.tan(0.95 * rad), 32, 20),
                                  new THREE.MeshBasicMaterial({ color: 0x4ea8de }));
scena.add(discoSole, discoTerra);

// Stelle
{
  const n = 1800, p = new Float32Array(n * 3);
  let s = 99;
  const casuale = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (let i = 0; i < n; i++) {
    const z = 2 * casuale() - 1, t = 2 * Math.PI * casuale(), r = Math.sqrt(1 - z * z);
    p[3 * i] = 8e5 * r * Math.cos(t); p[3 * i + 1] = 8e5 * z; p[3 * i + 2] = 8e5 * r * Math.sin(t);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(p, 3));
  var stelle = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xcfd6e6, size: 1.6, sizeAttenuation: false }));
  scena.add(stelle);
}

// Direzione nel cielo dato azimut (0 = nord, 90 = est) ed elevazione, in gradi
function direzioneCielo(az, el, out) {
  const a = az * rad, e = el * rad;
  return out.set(Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a));
}

// ---------- Personaggi (costruiti con forme semplici, rivolti verso -z) ----------
function materiale(colore, emissivo = 0x1c1c1c) {
  return new THREE.MeshLambertMaterial({ color: colore, emissive: emissivo });
}
function creaAstronauta() {
  const g = new THREE.Group(), bianco = materiale(0xe9e9e4), grigio = materiale(0x8d9096);
  const pezzo = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); g.add(m); return m; };
  pezzo(new THREE.CapsuleGeometry(0.27, 0.45, 6, 14), bianco, 0, 1.18, 0);                       // busto
  pezzo(new THREE.SphereGeometry(0.24, 20, 14), bianco, 0, 1.78, 0);                             // casco
  pezzo(new THREE.SphereGeometry(0.19, 20, 14), materiale(0xc79a2b, 0x3a2a05), 0, 1.78, -0.10);  // visiera
  pezzo(new THREE.BoxGeometry(0.46, 0.62, 0.24), grigio, 0, 1.25, 0.30);                         // zaino
  for (const lato of [-1, 1]) {
    pezzo(new THREE.CapsuleGeometry(0.10, 0.62, 4, 10), bianco, 0.14 * lato, 0.45, 0);           // gambe
    pezzo(new THREE.BoxGeometry(0.20, 0.10, 0.32), grigio, 0.14 * lato, 0.05, -0.04);            // stivali
    pezzo(new THREE.CapsuleGeometry(0.085, 0.50, 4, 10), bianco, 0.38 * lato, 1.20, 0).rotation.z = 0.18 * lato;  // braccia
  }
  return g;
}
function creaRover() {
  const g = new THREE.Group(), chiaro = materiale(0xd9d6cc), scuro = materiale(0x2b2d33);
  const pezzo = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); g.add(m); return m; };
  pezzo(new THREE.BoxGeometry(0.90, 0.28, 1.30), chiaro, 0, 0.55, 0);                            // telaio
  pezzo(new THREE.BoxGeometry(1.00, 0.03, 1.00), materiale(0x1d2b52, 0x070a16), 0, 0.74, 0.14);  // pannello solare
  pezzo(new THREE.CylinderGeometry(0.03, 0.03, 0.75, 8), scuro, 0.28, 1.06, -0.50);              // albero
  pezzo(new THREE.BoxGeometry(0.28, 0.12, 0.14), chiaro, 0.28, 1.47, -0.50);                     // testa con le camere
  pezzo(new THREE.CylinderGeometry(0.012, 0.012, 0.50, 6), scuro, -0.32, 1.0, 0.50);             // antenna
  pezzo(new THREE.ConeGeometry(0.15, 0.07, 18), materiale(0xb8902f, 0x2a1f05), -0.32, 1.27, 0.50);
  for (const lato of [-1, 1]) for (const z of [-0.5, 0, 0.5]) {
    pezzo(new THREE.CylinderGeometry(0.20, 0.20, 0.14, 16), scuro, 0.55 * lato, 0.20, z).rotation.z = Math.PI / 2;  // ruote
  }
  return g;
}
// occhi = altezza degli occhi in prima persona; mira e dist = punto guardato e distanza in terza persona
const PERSONAGGI = {
  astronauta: { modello: creaAstronauta(), occhi: 1.75, mira: 1.25, dist: 7 },
  rover: { modello: creaRover(), occhi: 1.45, mira: 0.75, dist: 6 },
};
for (const p of Object.values(PERSONAGGI)) { p.modello.matrixAutoUpdate = false; p.modello.visible = false; scena.add(p.modello); }

// Fascio luminoso che indica la destinazione scelta con un clic
const fascio = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 12, 1, true),
  new THREE.MeshBasicMaterial({ color: 0x6fd3ff, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
fascio.visible = false;
scena.add(fascio);

// ---------- Stato dell'osservatore ----------
const parametri = new URLSearchParams(location.search);
const numero = (nome, predefinito) => parametri.has(nome) ? parseFloat(parametri.get(nome)) : predefinito;
const stato = {
  e: numero("est", 0) * 1000, n: numero("nord", 0) * 1000,   // spostamento dalla partenza, in metri
  k: 0,                                                      // indice dell'istante scelto nell'intervallo
  dir: 0,                                                    // direzione dello sguardo
  incl: numero("incl", 0),                                   // inclinazione dello sguardo (prima persona)
  orb: numero("orb", 16),                                    // altezza angolare della camera (terza persona)
  dist: null,                                                // distanza della camera (terza persona)
  pers: parametri.get("pers") === "rover" ? "rover" : "astronauta",
  vista: parametri.get("vista") === "3" ? 3 : 1,
  alt: 0, rotta: 0,                                          // altezza degli occhi; direzione del personaggio
  vel: numero("vel", 1500),                                  // velocita' di spostamento, in metri al secondo
  viaggio: null,                                             // spostamento in corso dopo un clic
};
stato.alt = numero("alt", PERSONAGGI[stato.pers].occhi);

// Direzioni del Sole e della Terra: cambiano solo quando si sceglie un altro istante con il cursore
const versoSole = new THREE.Vector3(), versoTerra = new THREE.Vector3();
const gradi = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}°`;
function impostaIstante(k) {
  stato.k = Math.max(0, Math.min(D.cielo.length - 1, Math.round(k)));
  const c = D.cielo[stato.k];
  direzioneCielo(c.s[0], c.s[1], versoSole);
  direzioneCielo(c.e[0], c.e[1], versoTerra);
  luceSole.position.copy(versoSole).multiplyScalar(1e6);
  // Il disco solare ha un raggio di circa 0,27 gradi: la luce cala mentre scende sotto l'orizzontale
  luceSole.intensity = 3.2 * Math.max(0, Math.min(1, (c.s[1] + 0.27) / 0.54));
  discoTerra.visible = c.e[1] > -12;
  document.getElementById("cielo").textContent = `${c.t} (${D.fonte}) · Sole: azimut ${c.s[0].toFixed(1)}°, ` +
    `altezza ${gradi(c.s[1])} · Terra: azimut ${c.e[0].toFixed(1)}°, altezza ${gradi(c.e[1])}`;
  document.getElementById("ist").value = stato.k;
  document.getElementById("oIst").textContent = c.t.slice(5, 16);
}
// All'apertura: l'istante chiesto nell'indirizzo, altrimenti il primo con il Sole sopra l'orizzontale
const primoConLuce = D.cielo.findIndex((c) => c.s[1] > 0.3);
document.getElementById("ist").max = D.cielo.length - 1;
document.getElementById("rigaIstante").style.display = D.cielo.length > 1 ? "" : "none";
impostaIstante(numero("k", Math.max(0, primoConLuce)));
stato.dir = numero("dir", D.cielo[stato.k].s[1] > -1 ? D.cielo[stato.k].s[0] : 0);   // all'inizio si guarda verso il Sole
stato.rotta = stato.dir;

const ALT_MIN = 0.5, ALT_MAX = 5000, VEL_MIN = 20, VEL_MAX = 20000;   // cursori logaritmici
const daCursore = (t, min, max) => min * Math.pow(max / min, t);
const aCursore = (v, min, max) => Math.log(Math.min(Math.max(v, min), max) / min) / Math.log(max / min);
const metri = (v) => v < 1000 ? `${v < 10 ? v.toFixed(1) : Math.round(v)} m` : `${(v / 1000).toFixed(1)} km`;

const el = (id) => document.getElementById(id);
el("coord").textContent = `Lat ${D.lat.toFixed(3)}°, Lon ${D.lon.toFixed(3)}° · quota del terreno ${Math.round(D.quota0)} m`;
el("ist").addEventListener("input", (ev) => impostaIstante(+ev.target.value));
function aggiornaComandi() {
  el("alt").value = aCursore(stato.alt, ALT_MIN, ALT_MAX); el("oAlt").textContent = metri(stato.alt);
  el("alt").disabled = stato.vista === 3;
  el("vel").value = aCursore(stato.vel, VEL_MIN, VEL_MAX); el("oVel").textContent = `${metri(stato.vel)}/s`;
  el("vista1").classList.toggle("attivo", stato.vista === 1);
  el("vista3").classList.toggle("attivo", stato.vista === 3);
  el("persA").classList.toggle("attivo", stato.pers === "astronauta");
  el("persR").classList.toggle("attivo", stato.pers === "rover");
}
function impostaVista(v) { stato.vista = v; aggiornaComandi(); }
function impostaPersonaggio(nome) {
  stato.pers = nome; stato.alt = PERSONAGGI[nome].occhi; stato.dist = null; aggiornaComandi();
}
el("alt").addEventListener("input", (ev) => { stato.alt = daCursore(+ev.target.value, ALT_MIN, ALT_MAX); aggiornaComandi(); });
el("vel").addEventListener("input", (ev) => { stato.vel = daCursore(+ev.target.value, VEL_MIN, VEL_MAX); aggiornaComandi(); });
el("vista1").addEventListener("click", () => impostaVista(1));
el("vista3").addEventListener("click", () => impostaVista(3));
el("persA").addEventListener("click", () => impostaPersonaggio("astronauta"));
el("persR").addEventListener("click", () => impostaPersonaggio("rover"));
el("ritorna").addEventListener("click", () => { stato.e = 0; stato.n = 0; stato.viaggio = null; fascio.visible = false; });
aggiornaComandi();

// ---------- Mouse: trascinare ruota lo sguardo, un clic senza trascinare sceglie la destinazione ----------
const centroLuna = new THREE.Vector3(0, -(R + D.quota0), 0);
const raggioVisivo = new THREE.Raycaster(), puntatore = new THREE.Vector2(), appoggio = new THREE.Vector3();
let trascina = null;
tela.addEventListener("pointerdown", (ev) => {
  trascina = { x: ev.clientX, y: ev.clientY, percorso: 0 };
  tela.setPointerCapture(ev.pointerId);
});
tela.addEventListener("pointermove", (ev) => {
  if (!trascina) return;
  const dx = ev.clientX - trascina.x, dy = ev.clientY - trascina.y;
  trascina.x = ev.clientX; trascina.y = ev.clientY; trascina.percorso += Math.abs(dx) + Math.abs(dy);
  const gradiPerPixel = stato.vista === 1 ? camera.fov / tela.clientHeight : 0.25;
  stato.dir = (stato.dir - dx * gradiPerPixel + 360) % 360;
  if (stato.vista === 1) stato.incl = Math.max(-89, Math.min(89, stato.incl + dy * gradiPerPixel));
  else stato.orb = Math.max(2, Math.min(85, stato.orb + dy * gradiPerPixel));
});
tela.addEventListener("pointerup", (ev) => {
  const eraUnClic = trascina && trascina.percorso < 6;
  trascina = null;
  if (eraUnClic) vaiAlPuntoCliccato(ev);
});
tela.addEventListener("pointercancel", () => { trascina = null; });

function vaiAlPuntoCliccato(ev) {
  const r = tela.getBoundingClientRect();
  puntatore.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raggioVisivo.setFromCamera(puntatore, camera);
  const colpito = raggioVisivo.intersectObjects(terreni, false)[0];
  if (!colpito) return;                                   // clic sul cielo
  // Dal punto 3D colpito si torna alle coordinate (est, nord) misurate lungo la superficie
  const c = appoggio.copy(colpito.point).sub(centroLuna);
  const orizzontale = Math.hypot(c.x, c.z), d = R * Math.atan2(orizzontale, c.y);
  const e = orizzontale > 1e-9 ? d * c.x / orizzontale : 0, n = orizzontale > 1e-9 ? -d * c.z / orizzontale : 0;
  const tratto = Math.hypot(e - stato.e, n - stato.n);
  if (tratto < 1 || Math.hypot(e, n) >= 0.93 * D.raggio || quota(e, n) === null) return;
  stato.viaggio = { e0: stato.e, n0: stato.n, e1: e, n1: n, t: 0,
                    durata: Math.min(5, Math.max(1, tratto / (4 * stato.vel))) };
  stato.rotta = (Math.atan2(e - stato.e, n - stato.n) / rad + 360) % 360;
  // Fascio sulla destinazione, grande in proporzione alla distanza per restare visibile
  const hDest = quota(e, n), altezzaFascio = Math.max(8, tratto * 0.06);
  posizione3D(e, n, hDest + altezzaFascio / 2, fascio.position);
  appoggio.copy(fascio.position).sub(centroLuna).normalize();
  fascio.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), appoggio);
  fascio.scale.set(Math.max(0.15, tratto * 0.003), altezzaFascio, Math.max(0.15, tratto * 0.003));
  fascio.visible = true;
}

tela.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const f = ev.deltaY > 0 ? 1.1 : 0.9;
  if (stato.vista === 1) {
    camera.fov = Math.max(8, Math.min(90, camera.fov * f));
    camera.updateProjectionMatrix();
  } else {
    stato.dist = Math.max(2.5, Math.min(500, (stato.dist ?? PERSONAGGI[stato.pers].dist) * f));
  }
}, { passive: false });

// ---------- Tastiera ----------
const tasti = new Set();
addEventListener("keydown", (ev) => {
  if (ev.target instanceof HTMLInputElement) return;
  tasti.add(ev.code);
  if (ev.code.startsWith("Arrow")) ev.preventDefault();
  if (ev.code === "KeyV" && !ev.repeat) impostaVista(stato.vista === 1 ? 3 : 1);
  if (ev.code === "KeyC" && !ev.repeat) impostaPersonaggio(stato.pers === "astronauta" ? "rover" : "astronauta");
});
addEventListener("keyup", (ev) => tasti.delete(ev.code));
addEventListener("blur", () => tasti.clear());

function muovi(dt) {
  let avanti = 0, lato = 0;
  if (tasti.has("KeyW") || tasti.has("ArrowUp")) avanti += 1;
  if (tasti.has("KeyS") || tasti.has("ArrowDown")) avanti -= 1;
  if (tasti.has("KeyD") || tasti.has("ArrowRight")) lato += 1;
  if (tasti.has("KeyA") || tasti.has("ArrowLeft")) lato -= 1;
  if (avanti || lato) {
    stato.viaggio = null; fascio.visible = false;          // i tasti interrompono il viaggio verso il clic
    const spinta = tasti.has("ShiftLeft") || tasti.has("ShiftRight") ? 5 : 1;
    const passo = stato.vel * spinta * dt, a = stato.dir * rad;
    const de = avanti * Math.sin(a) + lato * Math.cos(a), dn = avanti * Math.cos(a) - lato * Math.sin(a);
    const e = stato.e + passo * de, n = stato.n + passo * dn;
    stato.rotta = (Math.atan2(de, dn) / rad + 360) % 360;
    if (Math.hypot(e, n) < 0.93 * D.raggio && quota(e, n) !== null) { stato.e = e; stato.n = n; }
  } else if (stato.viaggio) {
    const v = stato.viaggio;
    v.t = Math.min(1, v.t + dt / v.durata);
    const s = v.t * v.t * (3 - 2 * v.t);                   // partenza e arrivo morbidi
    stato.e = v.e0 + (v.e1 - v.e0) * s; stato.n = v.n0 + (v.n1 - v.n0) * s;
    if (v.t >= 1) { stato.viaggio = null; fascio.visible = false; }
  }
}

// ---------- Ciclo di disegno ----------
const suolo = new THREE.Vector3(), su = new THREE.Vector3(), est = new THREE.Vector3(), nord = new THREE.Vector3();
const tmp = new THREE.Vector3(), avantiV = new THREE.Vector3(), destra = new THREE.Vector3(), mira = new THREE.Vector3();
const normale = new THREE.Vector3(), appoggioRover = new THREE.Vector3(0, 1, 0);
const NOMI = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"];

// Perpendicolare al terreno sotto il personaggio, dalla pendenza misurata a un passo di distanza
function normaleTerreno(h, out) {
  const p = 0.6;
  const dhEst = ((quota(stato.e + p, stato.n) ?? h) - (quota(stato.e - p, stato.n) ?? h)) / (2 * p);
  const dhNord = ((quota(stato.e, stato.n + p) ?? h) - (quota(stato.e, stato.n - p) ?? h)) / (2 * p);
  return out.copy(su).addScaledVector(est, -dhEst).addScaledVector(nord, -dhNord).normalize();
}

function ridimensiona() {
  const w = tela.clientWidth, h = tela.clientHeight;
  if (tela.width !== Math.floor(w * renderer.getPixelRatio()) || tela.height !== Math.floor(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}
// direzione orizzontale per un dato azimut, negli assi locali
const orizzontale = (az, out) => out.set(0, 0, 0).addScaledVector(est, Math.sin(az * rad)).addScaledVector(nord, Math.cos(az * rad));

let precedente = performance.now();
function disegna(adesso) {
  const dt = Math.min(0.25, (adesso - precedente) / 1000);
  precedente = adesso;
  ridimensiona();
  muovi(dt);

  // Punto del suolo su cui ci si trova e assi locali (alto, est, nord)
  const h = quota(stato.e, stato.n) ?? D.quota0;
  posizione3D(stato.e, stato.n, h, suolo);
  su.copy(suolo).sub(centroLuna).normalize();
  posizione3D(stato.e + 10, stato.n, h, tmp);
  est.copy(tmp).sub(suolo);
  est.addScaledVector(su, -est.dot(su)).normalize();
  nord.crossVectors(su, est);

  const P = PERSONAGGI[stato.pers];
  for (const [nome, p] of Object.entries(PERSONAGGI)) p.modello.visible = stato.vista === 3 && nome === stato.pers;

  if (stato.vista === 1) {
    // Prima persona: la camera sta all'altezza degli occhi
    camera.position.copy(suolo).addScaledVector(su, stato.alt);
    orizzontale(stato.dir, avantiV).multiplyScalar(Math.cos(stato.incl * rad)).addScaledVector(su, Math.sin(stato.incl * rad));
    camera.up.copy(su);
    camera.lookAt(tmp.copy(camera.position).add(avantiV));
  } else {
    // Terza persona: il personaggio poggia sul suolo, girato nella direzione di marcia.
    // L'astronauta resta in piedi; il rover si inclina seguendo la pendenza, cosi' le ruote
    // restano appoggiate (il passaggio da una pendenza all'altra e' reso graduale).
    normaleTerreno(h, normale);
    appoggioRover.lerp(normale, 1 - Math.exp(-dt * 10)).normalize();
    const alto = stato.pers === "rover" ? appoggioRover : su;
    orizzontale(stato.rotta, avantiV);
    avantiV.addScaledVector(alto, -avantiV.dot(alto)).normalize();
    destra.crossVectors(avantiV, alto);
    P.modello.matrix.makeBasis(destra, alto, tmp.copy(avantiV).negate()).setPosition(suolo);
    // La camera gli gira intorno, a distanza e altezza regolabili
    const dist = stato.dist ?? P.dist, o = stato.orb * rad;
    mira.copy(suolo).addScaledVector(su, P.mira);
    orizzontale(stato.dir, avantiV);
    camera.position.copy(mira).addScaledVector(avantiV, -dist * Math.cos(o)).addScaledVector(su, dist * Math.sin(o));
    // ...senza finire sotto il terreno
    const ce = stato.e - Math.sin(stato.dir * rad) * dist * Math.cos(o), cn = stato.n - Math.cos(stato.dir * rad) * dist * Math.cos(o);
    const hc = quota(ce, cn);
    if (hc !== null) {
      const sotto = tmp.copy(camera.position).sub(posizione3D(ce, cn, hc + 0.4, destra)).dot(su);
      if (sotto < 0) camera.position.addScaledVector(su, -sotto);
    }
    camera.up.copy(su);
    camera.lookAt(mira);
  }

  // I dischi di Sole e Terra e le stelle seguono la camera, cosi' restano "all'infinito"
  discoSole.position.copy(camera.position).addScaledVector(versoSole, DIST_CIELO);
  discoTerra.position.copy(camera.position).addScaledVector(versoTerra, DIST_CIELO);
  stelle.position.copy(camera.position);

  el("bussola").innerHTML =
    `<b>${NOMI[Math.round(stato.dir / 45) % 8]} ${Math.round(stato.dir) % 360}°</b><br>` +
    (stato.vista === 1 ? `sguardo ${stato.incl >= 0 ? "+" : ""}${stato.incl.toFixed(1)}° · ` : "") +
    `${(Math.hypot(stato.e, stato.n) / 1000).toFixed(1)} km dalla partenza · suolo a ${Math.round(h)} m`;

  renderer.render(scena, camera);
  requestAnimationFrame(disegna);
}
requestAnimationFrame(disegna);
// Utile per controlli e prove dalla console del browser
window.__stato = stato;
window.__lab = { THREE, terreni, quota, posizione3D, centroLuna, D };
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
