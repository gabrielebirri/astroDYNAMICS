"""
"Street view" lunare: genera una pagina HTML in cui ci si guarda intorno in prima
persona dal sito scelto, ci si sposta sul terreno (tastiera o clic sulla destinazione),
si passa alla vista in terza persona con un astronauta o un rover.

Uso (horizon.py ed ephemeris.py devono stare nella stessa cartella):
    python streetview_lunare.py
    python streetview_lunare.py --sito 11 --istante "2026-11-27 00:00"
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
    """Da testo ("2026-11-27 00:00" o formato ISO) a data e ora UTC, al minuto."""
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
    p.add_argument("--istante", default="2026-12-01 05:00", help="un solo istante UTC, es. \"2026-11-27 00:00\" (predefinito: adesso)")
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
  #avviso { color: #f2c14e !important; }
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
  <p id="avviso"></p>
  <p>Trascina per guardarti intorno. Clic su un punto del terreno per andarci. W A S D o frecce per spostarti, Maiusc per andare più veloce. Rotella per lo zoom. V cambia vista, C cambia personaggio.</p>
</div>
<div id="comandi" class="riquadro">
  <label>Vista</label>
  <div class="scelta"><button id="vista1" type="button">Prima persona</button><button id="vista3" type="button">Terza persona</button></div>
  <label>Personaggio</label>
  <div class="scelta"><button id="persA" type="button">Astronauta</button><button id="persR" type="button">Rover</button></div>
  <label>Sassi e grana del suolo (decorativi)</label>
  <div class="scelta"><button id="detSi" type="button">Mostra</button><button id="detNo" type="button">Nascondi</button></div>
  <label>Velocità <output id="oVel"></output><input id="vel" type="range" min="0" max="1" step="0.001"></label>
  <label id="rigaIstante">Istante <output id="oIst"></output><input id="ist" type="range" min="0" step="1" value="0"></label>
  <label>Luminosità <output id="oLum"></output><input id="lum" type="range" min="0" max="1" step="0.001"></label>
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
// Resa dei toni simile a una macchina fotografica: le zone molto illuminate sfumano verso il bianco
renderer.toneMapping = THREE.ACESFilmicToneMapping;
// Ombre proiettate sul suolo dai sassi e dal personaggio (le ombre dei rilievi sono calcolate a parte)
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scena = new THREE.Scene();
scena.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(60, 1, 0.5, 1.2e6);

// ---------- Dettaglio decorativo del suolo ----------
// Il DEM ha un punto ogni 118 m: da vicino il terreno sarebbe liscio. Grana, piccoli crateri e
// sassi sono aggiunti solo per dare profondita' alla scena e NON provengono dai dati.
const LATO_GRANA = 24, LATO_TONI = 900;       // metri coperti da una ripetizione delle due texture

// Rumore che si ripete senza giunture ogni 'periodo' celle (serve a texture affiancabili)
function creaRumore(seme) {
  const tab = new Float32Array(65536);
  let s = seme;
  for (let i = 0; i < tab.length; i++) tab[i] = (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  return (x, y, periodo) => {
    const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    const x0 = ((xi % periodo) + periodo) % periodo, x1 = (x0 + 1) % periodo;
    const y0 = ((yi % periodo) + periodo) % periodo, y1 = (y0 + 1) % periodo;
    const a = tab[y0 * 256 + x0], b = tab[y0 * 256 + x1], c = tab[y1 * 256 + x0], d = tab[y1 * 256 + x1];
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
}
// Da una matrice di valori 0..1 a una texture in toni di grigio che si ripete
function creaTexture(lato, valori, colore) {
  const tela2 = document.createElement("canvas");
  tela2.width = tela2.height = lato;
  const ctx = tela2.getContext("2d"), img = ctx.createImageData(lato, lato);
  for (let i = 0; i < lato * lato; i++) {
    const v = Math.max(0, Math.min(255, Math.round(valori[i] * 255)));
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(tela2);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  if (colore) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
// Rilievo fine: ondulazioni, crateri da decimetri a un metro, sassolini
function creaGrana() {
  const L = 1024, alt = new Float32Array(L * L), rumore = creaRumore(4242);
  for (let y = 0; y < L; y++) for (let x = 0; x < L; x++) {
    let v = 0, amp = 0.5;
    for (const periodo of [6, 12, 24, 48, 96, 192]) { v += amp * rumore(x * periodo / L, y * periodo / L, periodo); amp *= 0.55; }
    alt[y * L + x] = 0.42 + 0.22 * v;
  }
  let s = 777;
  const casuale = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const timbro = (cx, cy, r, forma) => {            // applica una forma circolare, anche a cavallo dei bordi
    const R = Math.ceil(r * 1.6);
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const t = Math.hypot(dx, dy) / r;
      if (t >= 1.6) continue;
      const k = ((cy + dy + L) % L) * L + ((cx + dx + L) % L);
      alt[k] += forma(t);
    }
  };
  for (let i = 0; i < 90; i++) {                    // crateri: conca con bordo rialzato
    const r = 5 + 40 * Math.pow(casuale(), 3), prof = 0.10 + 0.12 * casuale();
    timbro(Math.floor(casuale() * L), Math.floor(casuale() * L), r,
           (t) => t < 1 ? -prof * (1 - t * t) + 0.3 * prof * t * t : 0.3 * prof * (1 - (t - 1) / 0.6));
  }
  for (let i = 0; i < 900; i++) {                   // sassolini: piccole cupole
    const r = 1.5 + 4 * casuale() * casuale(), h = 0.10 + 0.15 * casuale();
    timbro(Math.floor(casuale() * L), Math.floor(casuale() * L), r, (t) => t < 1 ? h * Math.sqrt(1 - t * t) : 0);
  }
  return creaTexture(L, alt, false);
}
// Variazioni di tono su scala di centinaia di metri
function creaToni() {
  const L = 512, v = new Float32Array(L * L), rumore = creaRumore(99);
  for (let y = 0; y < L; y++) for (let x = 0; x < L; x++) {
    let t = 0, amp = 0.5;
    for (const periodo of [3, 7, 16, 40, 100]) { t += amp * rumore(x * periodo / L, y * periodo / L, periodo); amp *= 0.6; }
    v[y * L + x] = 0.78 + 0.22 * t;
  }
  return creaTexture(L, v, true);
}
const texGrana = creaGrana(), texToni = creaToni();
texGrana.channel = 1;                              // usa il secondo insieme di coordinate (uv1)
const RILIEVO_GRANA = 3;
const materialeSuolo = new THREE.MeshStandardMaterial({
  vertexColors: true, roughness: 1, metalness: 0, map: texToni, bumpMap: texGrana, bumpScale: RILIEVO_GRANA,
});
// La grana si vede solo vicino all'osservatore: oltre qualche decina di metri sfuma fino a sparire,
// altrimenti in lontananza produrrebbe un fastidioso effetto a puntini
materialeSuolo.onBeforeCompile = (shader) => {
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <normal_fragment_maps>",
    THREE.ShaderChunk.normal_fragment_maps.replace(
      "dHdxy_fwd()", "( dHdxy_fwd() * ( 1.0 - smoothstep( 12.0, 60.0, length( vViewPosition ) ) ) )"));
};

// Costruisce la superficie di una griglia. 'escludi' e' il mezzo lato (in metri) della
// zona centrale da lasciare vuota, perche' coperta dalla griglia piu' dettagliata.
function creaSuperficie(g, escludi, rumore) {
  const N = g.n, pos = new Float32Array(N * N * 3), col = new Float32Array(N * N * 3);
  const uv = new Float32Array(N * N * 2), uv1 = new Float32Array(N * N * 2);
  const valido = new Uint8Array(N * N), v = new THREE.Vector3();
  g.albedo = new Float32Array(N * N);              // tono del suolo in piena luce
  g.ombra = new Float32Array(N * N).fill(1);       // frazione di Sole visibile (1 = luce, 0 = ombra)
  g.Y = new Float32Array(N * N).fill(NaN);         // quota lungo la verticale del sito, curvatura inclusa
  let seme = 12345;
  const casuale = () => (seme = (seme * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const k = j * N + i, raw = g.h[k];
    if (raw === SENZA_DATO) continue;
    valido[k] = 1;
    const e = i * g.passo - g.meta, n = j * g.passo - g.meta;
    posizione3D(e, n, raw * 0.5, v);
    pos[3 * k] = v.x; pos[3 * k + 1] = v.y; pos[3 * k + 2] = v.z;
    g.Y[k] = v.y;
    g.albedo[k] = 0.58 + rumore * (casuale() - 0.5);
    col[3 * k] = col[3 * k + 1] = col[3 * k + 2] = g.albedo[k];
    uv[2 * k] = e / LATO_TONI; uv[2 * k + 1] = n / LATO_TONI;
    uv1[2 * k] = e / LATO_GRANA; uv1[2 * k + 1] = n / LATO_GRANA;
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
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setAttribute("uv1", new THREE.BufferAttribute(uv1, 2));
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indici), 1));
  geo.computeVertexNormals();
  g.geo = geo; g.valido = valido;
  const mesh = new THREE.Mesh(geo, materialeSuolo);
  mesh.receiveShadow = true;
  return mesh;
}
const terreni = [creaSuperficie(D.vicino, 0, 0.06), creaSuperficie(D.lontano, D.vicino.meta, 0.03)];
scena.add(...terreni);

// Luce: il Sole come unica sorgente, piu' un filo di luce diffusa per non avere nero assoluto
const luceSole = new THREE.DirectionalLight(0xffffff, 3.2);
luceSole.castShadow = true;                        // ombre di sassi e personaggio, attorno all'osservatore
luceSole.shadow.mapSize.set(2048, 2048);
Object.assign(luceSole.shadow.camera, { left: -45, right: 45, top: 45, bottom: -45, near: 10, far: 6000 });
luceSole.shadow.camera.updateProjectionMatrix();
luceSole.shadow.bias = -0.0003;
luceSole.shadow.normalBias = 0.02;
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

// ---------- Personaggi (costruiti con forme semplici, rivolti verso -z, origine al suolo) ----------
// Una parte del colore e' sempre visibile (emissione), cosi' il personaggio si distingue anche in controluce
function materiale(colore) {
  const m = new THREE.MeshLambertMaterial({ color: colore, emissive: new THREE.Color(colore).multiplyScalar(0.3) });
  m.userData.base = new THREE.Color(colore);
  return m;
}
// Aggiunge un pezzo a un gruppo; restituisce la mesh per eventuali ritocchi
function pezzo(gruppo, geo, mat, x, y, z) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z); m.castShadow = true;
  gruppo.add(m);
  return m;
}
// Cilindro sottile tra due punti (aste, tubi delle sospensioni)
function tubo(gruppo, a, b, raggio, mat) {
  const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b), asse = B.clone().sub(A);
  const m = pezzo(gruppo, new THREE.CylinderGeometry(raggio, raggio, asse.length(), 10), mat, 0, 0, 0);
  m.position.copy(A).addScaledVector(asse, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), asse.normalize());
  return m;
}

// Astronauta con tuta per attivita' extraveicolare: busto rigido, zaino di supporto vitale,
// casco con visiera dorata, anelli alle giunture, guanti, stivali e fasce rosse di riconoscimento
function creaAstronauta() {
  const g = new THREE.Group();
  const bianco = materiale(0xe8e8e2), grigio = materiale(0x8f9299), scuro = materiale(0x2c2e33);
  const rosso = materiale(0xb3261e), oro = materiale(0xc9a033), blu = materiale(0x2a5c9a);
  const anello = (gruppo, r, y) => { const a = pezzo(gruppo, new THREE.TorusGeometry(r, 0.022, 8, 20), grigio, 0, y, 0); a.rotation.x = Math.PI / 2; return a; };

  pezzo(g, new THREE.CapsuleGeometry(0.26, 0.30, 6, 16), bianco, 0, 1.24, 0).scale.z = 0.86;          // busto rigido
  pezzo(g, new THREE.SphereGeometry(0.25, 18, 12), bianco, 0, 0.94, 0).scale.set(1, 0.72, 0.86);      // bacino
  anello(g, 0.235, 1.02);                                                                              // anello in vita
  pezzo(g, new THREE.BoxGeometry(0.52, 0.74, 0.26), bianco, 0, 1.32, 0.30);                           // zaino di supporto vitale
  pezzo(g, new THREE.BoxGeometry(0.44, 0.10, 0.22), grigio, 0, 1.74, 0.30);
  pezzo(g, new THREE.BoxGeometry(0.30, 0.34, 0.02), grigio, 0, 1.30, 0.44);
  pezzo(g, new THREE.BoxGeometry(0.28, 0.20, 0.10), grigio, 0, 1.27, -0.25);                          // pannello di controllo sul petto
  pezzo(g, new THREE.BoxGeometry(0.05, 0.05, 0.03), rosso, -0.08, 1.31, -0.31);
  pezzo(g, new THREE.BoxGeometry(0.05, 0.05, 0.03), oro, 0.0, 1.31, -0.31);
  pezzo(g, new THREE.BoxGeometry(0.05, 0.05, 0.03), scuro, 0.08, 1.31, -0.31);
  pezzo(g, new THREE.BoxGeometry(0.10, 0.07, 0.01), blu, 0.15, 1.45, -0.235);                         // toppa sul petto
  anello(g, 0.20, 1.58);                                                                               // collare del casco
  pezzo(g, new THREE.SphereGeometry(0.25, 24, 16), bianco, 0, 1.80, 0);                               // casco
  pezzo(g, new THREE.SphereGeometry(0.215, 24, 16), oro, 0, 1.80, -0.075).scale.y = 0.9;              // visiera dorata
  for (const lato of [-1, 1]) pezzo(g, new THREE.BoxGeometry(0.05, 0.09, 0.10), grigio, 0.255 * lato, 1.84, -0.03);   // luci del casco

  const arti = { braccia: [], gambe: [] };
  for (const lato of [-1, 1]) {
    const braccio = new THREE.Group();                                                                 // ruota attorno alla spalla
    braccio.position.set(0.335 * lato, 1.48, 0); braccio.rotation.z = 0.10 * lato;
    pezzo(braccio, new THREE.SphereGeometry(0.115, 14, 10), bianco, 0, 0, 0);
    pezzo(braccio, new THREE.CapsuleGeometry(0.085, 0.20, 4, 10), bianco, 0, -0.20, 0);
    pezzo(braccio, new THREE.CylinderGeometry(0.092, 0.092, 0.04, 14), rosso, 0, -0.22, 0);           // fascia rossa
    anello(braccio, 0.088, -0.37);
    pezzo(braccio, new THREE.CapsuleGeometry(0.078, 0.18, 4, 10), bianco, 0, -0.52, 0);
    anello(braccio, 0.08, -0.66);
    pezzo(braccio, new THREE.SphereGeometry(0.085, 12, 10), grigio, 0, -0.73, 0).scale.set(0.9, 1.1, 1);   // guanto
    g.add(braccio); arti.braccia.push(braccio);

    const gamba = new THREE.Group();                                                                   // ruota attorno all'anca
    gamba.position.set(0.135 * lato, 0.88, 0);
    pezzo(gamba, new THREE.CapsuleGeometry(0.105, 0.20, 4, 10), bianco, 0, -0.20, 0);
    pezzo(gamba, new THREE.CylinderGeometry(0.112, 0.112, 0.04, 14), rosso, 0, -0.16, 0);
    anello(gamba, 0.10, -0.38);
    pezzo(gamba, new THREE.CapsuleGeometry(0.095, 0.20, 4, 10), bianco, 0, -0.56, 0);
    pezzo(gamba, new THREE.BoxGeometry(0.20, 0.14, 0.34), grigio, 0, -0.78, -0.05);                   // stivale
    pezzo(gamba, new THREE.BoxGeometry(0.21, 0.04, 0.36), scuro, 0, -0.86, -0.05);
    g.add(gamba); arti.gambe.push(gamba);
  }
  g.userData.arti = arti;
  return g;
}

// Rover a sei ruote sul modello di Perseverance: sospensioni "rocker-bogie", albero con le camere,
// braccio robotico ripiegato davanti, generatore a radioisotopi dietro, antenne sul ponte
function creaRover() {
  const g = new THREE.Group();
  const bianco = materiale(0xdcdad2), nero = materiale(0x1e1f23), titanio = materiale(0x4a4c52);
  const alluminio = materiale(0x9ea2a8), ruotaMat = materiale(0x6a6d73), oro = materiale(0xb08a2e), vetro = materiale(0x101820);

  pezzo(g, new THREE.BoxGeometry(1.55, 0.50, 2.10), bianco, 0, 0.95, 0);                              // corpo
  pezzo(g, new THREE.BoxGeometry(1.62, 0.04, 2.16), alluminio, 0, 1.22, 0);                           // ponte superiore
  pezzo(g, new THREE.BoxGeometry(1.50, 0.03, 2.05), nero, 0, 0.69, 0);                                // fondo
  pezzo(g, new THREE.BoxGeometry(0.50, 0.14, 0.40), alluminio, -0.35, 1.31, -0.45);                   // apparati sul ponte
  pezzo(g, new THREE.BoxGeometry(0.34, 0.10, 0.50), oro, 0.30, 1.29, 0.20);
  pezzo(g, new THREE.BoxGeometry(0.22, 0.18, 0.22), bianco, -0.50, 1.33, 0.75);

  // Sospensioni e ruote: per lato un bilanciere (ruota anteriore) e un carrello (ruote centrale e posteriore)
  const ruote = [], Xr = 1.12, Yr = 0.2625, zRuote = [-1.25, 0.05, 1.15];
  const geoRuota = new THREE.CylinderGeometry(Yr, Yr, 0.38, 28); geoRuota.rotateZ(Math.PI / 2);
  const geoMozzo = new THREE.CylinderGeometry(0.09, 0.09, 0.40, 14); geoMozzo.rotateZ(Math.PI / 2);
  const geoCostola = new THREE.BoxGeometry(0.39, 0.035, 0.05);
  for (const lato of [-1, 1]) {
    const x = Xr * lato, perno = [0.80 * lato, 0.88, -0.10], carrello = [x, 0.56, 0.62];
    tubo(g, perno, [x, 0.62, -0.75], 0.045, titanio);                                                  // bilanciere, tratto anteriore
    tubo(g, [x, 0.62, -0.75], [x, 0.50, zRuote[0]], 0.04, titanio);
    tubo(g, perno, carrello, 0.045, titanio);                                                          // bilanciere, tratto posteriore
    tubo(g, carrello, [x, 0.50, zRuote[1]], 0.04, titanio);                                            // carrello
    tubo(g, carrello, [x, 0.50, zRuote[2]], 0.04, titanio);
    pezzo(g, new THREE.SphereGeometry(0.07, 10, 8), nero, ...carrello);
    for (const z of zRuote) {
      tubo(g, [x, 0.50, z], [x - 0.12 * lato, Yr, z], 0.035, titanio);                                 // montante della ruota
      const ruota = new THREE.Group();
      ruota.position.set(x, Yr, z);
      pezzo(ruota, geoRuota, ruotaMat, 0, 0, 0);
      pezzo(ruota, geoMozzo, nero, 0, 0, 0);
      for (let k = 0; k < 12; k++) {                                                                   // costole del battistrada
        const a = k * Math.PI / 6, c = pezzo(ruota, geoCostola, nero, 0, (Yr + 0.008) * Math.cos(a), (Yr + 0.008) * Math.sin(a));
        c.rotation.x = a;
      }
      g.add(ruota); ruote.push(ruota);
    }
  }
  tubo(g, [-0.80, 1.27, -0.10], [0.80, 1.27, -0.10], 0.03, nero);                                      // barra del differenziale

  // Albero con la testa delle camere (davanti a destra)
  tubo(g, [0.55, 1.24, -0.85], [0.55, 2.02, -0.85], 0.05, bianco);
  pezzo(g, new THREE.BoxGeometry(0.44, 0.24, 0.24), bianco, 0.55, 2.12, -0.87);
  const occhio = pezzo(g, new THREE.CylinderGeometry(0.075, 0.075, 0.06, 18), vetro, 0.66, 2.14, -1.0);   // strumento laser
  occhio.rotation.x = Math.PI / 2;
  for (const dx of [-0.10, 0.04]) pezzo(g, new THREE.BoxGeometry(0.06, 0.06, 0.05), vetro, 0.48 + dx, 2.10, -1.0);   // camere stereo

  // Braccio robotico ripiegato sul davanti, con la torretta degli strumenti
  pezzo(g, new THREE.CylinderGeometry(0.09, 0.09, 0.16, 14), alluminio, -0.55, 0.98, -1.10);
  tubo(g, [-0.55, 1.00, -1.14], [0.48, 1.00, -1.22], 0.05, titanio);
  tubo(g, [0.48, 1.00, -1.22], [-0.30, 0.84, -1.34], 0.045, titanio);
  pezzo(g, new THREE.SphereGeometry(0.075, 10, 8), nero, 0.48, 1.00, -1.22);
  const torretta = pezzo(g, new THREE.CylinderGeometry(0.19, 0.19, 0.20, 16), alluminio, -0.36, 0.82, -1.42);
  torretta.rotation.x = Math.PI / 2;
  pezzo(g, new THREE.BoxGeometry(0.18, 0.18, 0.14), nero, -0.36, 0.82, -1.56);

  // Generatore a radioisotopi sul retro, inclinato, con le alette di raffreddamento
  const generatore = new THREE.Group();
  generatore.position.set(0, 1.18, 1.22); generatore.rotation.x = -0.45;
  pezzo(generatore, new THREE.CylinderGeometry(0.20, 0.20, 0.66, 16), bianco, 0, 0.30, 0);
  for (let k = 0; k < 8; k++) pezzo(generatore, new THREE.BoxGeometry(0.02, 0.60, 0.62), nero, 0, 0.30, 0).rotation.y = k * Math.PI / 8;
  g.add(generatore);

  // Antenne: ad alto guadagno (esagonale, orientabile), a basso guadagno e UHF
  tubo(g, [-0.45, 1.24, 0.30], [-0.45, 1.50, 0.30], 0.03, alluminio);
  const piatto = pezzo(g, new THREE.CylinderGeometry(0.30, 0.30, 0.03, 6), oro, -0.45, 1.56, 0.30);
  piatto.rotation.x = 0.5;
  tubo(g, [0.52, 1.24, 0.80], [0.52, 1.62, 0.80], 0.05, alluminio);
  tubo(g, [-0.62, 1.24, -0.10], [-0.62, 1.44, -0.10], 0.025, alluminio);

  g.userData.ruote = ruote;
  return g;
}
// occhi = altezza degli occhi in prima persona; mira e dist = punto guardato e distanza in terza persona;
// passo = distanza a cui si misura la pendenza del terreno sotto il personaggio
const PERSONAGGI = {
  astronauta: { modello: creaAstronauta(), occhi: 1.75, mira: 1.25, dist: 7, passo: 0.6 },
  rover: { modello: creaRover(), occhi: 2.1, mira: 1.1, dist: 10, passo: 1.3 },
};
for (const p of Object.values(PERSONAGGI)) {
  p.modello.matrixAutoUpdate = false; p.modello.visible = false; scena.add(p.modello);
  p.materiali = new Set();
  p.modello.traverse((o) => { if (o.isMesh) p.materiali.add(o.material); });
}

// ---------- Ombre portate dai rilievi (stesso metodo di ephemeris.py) ----------
// Per ogni punto del terreno si percorre il suolo verso il Sole e si cerca l'angolo massimo sotto cui
// appaiono i rilievi in quella direzione. Se supera l'altezza del Sole il punto e' in ombra; se taglia
// il disco solare (raggio 0,2666 gradi) e' in penombra. Il risultato scurisce il colore del terreno.
const RAGGIO_SOLE = 0.2666;
function quotaY(e, n) {                               // quota lungo la verticale del sito; -Infinity senza dati
  let g = D.vicino;
  if (Math.abs(e) > g.meta || Math.abs(n) > g.meta) g = D.lontano;
  const x = (e + g.meta) / g.passo, y = (n + g.meta) / g.passo;
  if (x < 0 || y < 0 || x > g.n - 1 || y > g.n - 1) return -Infinity;
  const i = Math.min(Math.floor(x), g.n - 2), j = Math.min(Math.floor(y), g.n - 2), fx = x - i, fy = y - j;
  const k = j * g.n + i, v = g.Y[k] * (1 - fx) * (1 - fy) + g.Y[k + 1] * fx * (1 - fy) + g.Y[k + g.n] * (1 - fx) * fy + g.Y[k + g.n + 1] * fx * fy;
  return v === v ? v : -Infinity;
}
function distanzeDiMarcia(passo, dMax) {              // fitte vicino, poi crescono del 2 per cento
  const d = [];
  for (let k = 1; k <= 40; k++) d.push(k * passo);
  while (d[d.length - 1] < dMax) d.push(d[d.length - 1] + Math.max(passo, 0.02 * d[d.length - 1]));
  return Float64Array.from(d);
}
function frazioneDisco(el, orizzonte) {               // parte del disco solare sopra il profilo dei rilievi
  const x = Math.max(-1, Math.min(1, (el - orizzonte) / RAGGIO_SOLE));
  return 1 - (Math.acos(x) - x * Math.sqrt(1 - x * x)) / Math.PI;
}
let quotaMassima = -Infinity;
for (const g of [D.vicino, D.lontano]) {
  g.marcia = distanzeDiMarcia(g.passo, 2 * D.raggio);
  for (let k = 0; k < g.Y.length; k++) if (g.Y[k] > quotaMassima) quotaMassima = g.Y[k];
}
function calcolaOmbre() {
  const c = D.cielo[stato.k], az = c.s[0] * rad, el = c.s[1];
  const se = Math.sin(az), sn = Math.cos(az);
  const tanAlto = Math.tan((el + RAGGIO_SOLE) * rad), tanBasso = Math.tan((el - RAGGIO_SOLE) * rad);
  for (const g of [D.vicino, D.lontano]) {
    const N = g.n, dist = g.marcia, interno = g === D.lontano ? D.vicino.meta - g.passo : -1;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const k = j * N + i;
      if (!g.valido[k]) continue;
      const e0 = i * g.passo - g.meta, n0 = j * g.passo - g.meta;
      if (Math.abs(e0) < interno && Math.abs(n0) < interno) continue;     // coperto dalla griglia vicina
      const Y0 = g.Y[k];
      let tanMax = -Infinity;
      for (let q = 0; q < dist.length; q++) {
        const d = dist[q];
        if (tanBasso > 0 && quotaMassima - Y0 < tanBasso * d) break;      // piu' in la' nessun rilievo arriva al Sole
        const Yt = quotaY(e0 + d * se, n0 + d * sn);
        if (Yt === -Infinity) break;                                      // fine del terreno disponibile
        const t = (Yt - Y0) / d;
        if (t > tanMax) { tanMax = t; if (tanMax >= tanAlto) break; }     // disco solare gia' tutto coperto
      }
      g.ombra[k] = frazioneDisco(el, Math.atan(tanMax) / rad);
    }
    const col = g.geo.attributes.color;
    for (let k = 0; k < g.ombra.length; k++) {
      const v = g.albedo[k] * (0.02 + 0.98 * g.ombra[k]);
      col.array[3 * k] = col.array[3 * k + 1] = col.array[3 * k + 2] = v;
    }
    col.needsUpdate = true;
  }
  ombreCalcolate = true;
  for (const S of strati) S.ci = null;                                    // i sassi vanno ricolorati
}
let ombreCalcolate = false, attesaOmbre = 0;
function richiediOmbre() {                            // il calcolo dura circa un secondo: prima si mostra l'avviso
  document.getElementById("avviso").textContent = "Calcolo delle ombre dei rilievi…";
  clearTimeout(attesaOmbre);
  attesaOmbre = setTimeout(() => {
    const t0 = performance.now();
    calcolaOmbre();
    window.__tempoOmbre = performance.now() - t0;
    document.getElementById("avviso").textContent = "";
  }, 60);
}
// Frazione di Sole visibile in un punto qualunque (interpolata tra i punti della griglia)
function luceIn(e, n) {
  let g = D.vicino;
  if (Math.abs(e) > g.meta || Math.abs(n) > g.meta) g = D.lontano;
  const x = (e + g.meta) / g.passo, y = (n + g.meta) / g.passo;
  if (x < 0 || y < 0 || x > g.n - 1 || y > g.n - 1) return 1;
  const i = Math.min(Math.floor(x), g.n - 2), j = Math.min(Math.floor(y), g.n - 2), fx = x - i, fy = y - j, k = j * g.n + i;
  return g.ombra[k] * (1 - fx) * (1 - fy) + g.ombra[k + 1] * fx * (1 - fy) + g.ombra[k + g.n] * (1 - fx) * fy + g.ombra[k + g.n + 1] * fx * fy;
}

// Luce che arriva a un oggetto appoggiato al suolo: niente se il punto e' nell'ombra di un rilievo
// oppure se il pendio su cui poggia e' girato dalla parte opposta al Sole
const soleENU = [0, 0, 1];                            // direzione del Sole (est, nord, alto), aggiornata con l'istante
function luceSuolo(e, n) {
  const p = 2, h0 = quota(e, n);
  if (h0 === null) return 0;
  const dhE = ((quota(e + p, n) ?? h0) - (quota(e - p, n) ?? h0)) / (2 * p);
  const dhN = ((quota(e, n + p) ?? h0) - (quota(e, n - p) ?? h0)) / (2 * p);
  const incidenza = (-dhE * soleENU[0] - dhN * soleENU[1] + soleENU[2]) / Math.hypot(dhE, dhN, 1);
  return luceIn(e, n) * Math.max(0, Math.min(1, incidenza / 0.06));
}

// ---------- Sassi e massi (decorativi) ----------
// La posizione di ogni sasso dipende solo dalla cella di terreno in cui cade, quindi i sassi restano
// fermi mentre ci si sposta. Vengono creati solo quelli attorno all'osservatore.
function caso(ix, iz, k) {                            // numero pseudo-casuale tra 0 e 1, fisso per ogni cella
  let h = (Math.imul(ix, 374761393) + Math.imul(iz, 668265263) + Math.imul(k, 1442695041)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
const geoSasso = new THREE.IcosahedronGeometry(1, 1);
{                                                     // forma irregolare: ogni vertice e' spostato lungo il raggio
  const p = geoSasso.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const f = 0.72 + 0.5 * caso(Math.round(x * 97), Math.round(y * 89 + z * 83), 7);
    p.setXYZ(i, x * f, y * f, z * f);
  }
  geoSasso.computeVertexNormals();
}
const matSasso = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0, flatShading: true });
function creaStrato(cella, raggio, prob, sMin, sMax, capienza) {
  const mesh = new THREE.InstancedMesh(geoSasso, matSasso, capienza);
  mesh.castShadow = true; mesh.frustumCulled = false; mesh.count = 0;
  scena.add(mesh);
  return { mesh, cella, raggio, prob, sMin, sMax, capienza, ci: null, cj: null };
}
const strati = [
  creaStrato(8.0, 150, 0.11, 0.25, 2.2, 1200),        // massi radi, visibili da lontano
  creaStrato(1.3, 30, 0.42, 0.04, 0.22, 1400),        // sassi fitti, solo vicino
];
const mSasso = new THREE.Matrix4(), pSasso = new THREE.Vector3(), qSasso = new THREE.Quaternion();
const sSasso = new THREE.Vector3(), eSasso = new THREE.Euler(), cSasso = new THREE.Color();
function aggiornaStrato(S) {
  const ci = Math.floor(stato.e / S.cella), cj = Math.floor(stato.n / S.cella);
  if (ci === S.ci && cj === S.cj) return;             // l'osservatore e' ancora nella stessa cella
  S.ci = ci; S.cj = cj;
  const m = Math.ceil(S.raggio / S.cella);
  let k = 0;
  for (let dj = -m; dj <= m && k < S.capienza; dj++) for (let di = -m; di <= m && k < S.capienza; di++) {
    const ix = ci + di, iz = cj + dj;
    if (caso(ix, iz, 1) > S.prob) continue;
    const e = (ix + caso(ix, iz, 2)) * S.cella, n = (iz + caso(ix, iz, 3)) * S.cella;
    if (Math.hypot(e - stato.e, n - stato.n) > S.raggio) continue;
    const h = quota(e, n);
    if (h === null) continue;
    const u = caso(ix, iz, 4), s = S.sMin * Math.pow(S.sMax / S.sMin, u * u * u);   // molti piccoli, pochi grandi
    sSasso.set(s * (0.7 + 0.6 * caso(ix, iz, 5)), s * (0.42 + 0.4 * caso(ix, iz, 6)), s * (0.7 + 0.6 * caso(ix, iz, 8)));
    posizione3D(e, n, h + 0.25 * sSasso.y, pSasso);   // in parte affondato nel suolo
    eSasso.set(0.5 * (caso(ix, iz, 9) - 0.5), 6.2832 * caso(ix, iz, 10), 0.5 * (caso(ix, iz, 11) - 0.5));
    S.mesh.setMatrixAt(k, mSasso.compose(pSasso, qSasso.setFromEuler(eSasso), sSasso));
    const tono = (0.40 + 0.28 * caso(ix, iz, 12)) * (0.03 + 0.97 * luceSuolo(e, n));  // scuro se in ombra
    S.mesh.setColorAt(k, cSasso.setRGB(tono, tono, tono * 0.98));
    k++;
  }
  S.mesh.count = k;
  S.mesh.instanceMatrix.needsUpdate = true;
  if (S.mesh.instanceColor) S.mesh.instanceColor.needsUpdate = true;
}

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
  dist: parametri.has("dist") ? parseFloat(parametri.get("dist")) : null,   // distanza della camera (terza persona)
  pers: parametri.get("pers") === "rover" ? "rover" : "astronauta",
  vista: parametri.get("vista") === "3" ? 3 : 1,
  dettagli: parametri.get("det") !== "0",                    // sassi e grana decorativi
  inMoto: false, fase: 0,                                    // per animare il passo e le ruote
  alt: 0, rotta: 0,                                          // altezza degli occhi; direzione del personaggio
  vel: numero("vel", 1500),                                  // velocita' di spostamento, in metri al secondo
  lum: numero("lum", 1),                                     // correzione manuale della luminosita' (1 = automatica)
  viaggio: null,                                             // spostamento in corso dopo un clic
};
stato.alt = numero("alt", PERSONAGGI[stato.pers].occhi);

// Direzioni del Sole e della Terra: cambiano solo quando si sceglie un altro istante con il cursore
const versoSole = new THREE.Vector3(), versoTerra = new THREE.Vector3();
let comandiPronti = false;                                   // diventa vero quando i comandi della pagina sono collegati
const gradi = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}°`;
function impostaIstante(k) {
  stato.k = Math.max(0, Math.min(D.cielo.length - 1, Math.round(k)));
  const c = D.cielo[stato.k];
  direzioneCielo(c.s[0], c.s[1], versoSole);
  direzioneCielo(c.e[0], c.e[1], versoTerra);
  soleENU[0] = Math.sin(c.s[0] * rad) * Math.cos(c.s[1] * rad);
  soleENU[1] = Math.cos(c.s[0] * rad) * Math.cos(c.s[1] * rad);
  soleENU[2] = Math.sin(c.s[1] * rad);
  richiediOmbre();
  if (comandiPronti) aggiornaComandi();
  // Esposizione automatica, come in una fotocamera: con il Sole radente il suolo piano riceve
  // pochissima luce (proporzionale al seno dell'altezza del Sole), quindi la si compensa in modo
  // che il terreno pianeggiante abbia sempre una luminosita' simile. I pendii rivolti al Sole
  // risultano molto chiari, le zone in ombra restano nere.
  const compensazione = Math.min(120, 1.7 / Math.sin(Math.max(c.s[1], 0.5) * rad));
  // Il disco solare ha un raggio di circa 0,27 gradi: la luce cala mentre scende sotto l'orizzontale
  const sopra = Math.max(0, Math.min(1, (c.s[1] + 0.27) / 0.54));
  luceSole.intensity = compensazione * sopra;
  const notte = sopra === 0 ? " · NOTTE: il Sole è sotto l'orizzonte, la scena è buia" : "";
  discoTerra.visible = c.e[1] > -12;
  document.getElementById("cielo").textContent = `${c.t} (${D.fonte}) · Sole: azimut ${c.s[0].toFixed(1)}°, ` +
    `altezza ${gradi(c.s[1])} · Terra: azimut ${c.e[0].toFixed(1)}°, altezza ${gradi(c.e[1])}${notte}`;
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

const ALT_MIN = 0.5, ALT_MAX = 5000, VEL_MIN = 20, VEL_MAX = 20000, LUM_MIN = 0.2, LUM_MAX = 5;   // cursori logaritmici
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
  el("lum").value = aCursore(stato.lum, LUM_MIN, LUM_MAX); el("oLum").textContent = `× ${stato.lum.toFixed(2)}`;
  renderer.toneMappingExposure = stato.lum;
  el("vista1").classList.toggle("attivo", stato.vista === 1);
  el("vista3").classList.toggle("attivo", stato.vista === 3);
  el("persA").classList.toggle("attivo", stato.pers === "astronauta");
  el("persR").classList.toggle("attivo", stato.pers === "rover");
  el("detSi").classList.toggle("attivo", stato.dettagli);
  el("detNo").classList.toggle("attivo", !stato.dettagli);
  // Con il Sole radente la grana viene attenuata, altrimenti il suolo si riempie di riflessi
  const altezzaSole = D.cielo[stato.k].s[1];
  materialeSuolo.bumpScale = stato.dettagli ? RILIEVO_GRANA * Math.max(0.3, Math.min(1, altezzaSole / 15)) : 0;
  for (const S of strati) S.mesh.visible = stato.dettagli;
}
function impostaVista(v) { stato.vista = v; aggiornaComandi(); }
function impostaPersonaggio(nome) {
  stato.pers = nome; stato.alt = PERSONAGGI[nome].occhi; stato.dist = null; aggiornaComandi();
}
el("alt").addEventListener("input", (ev) => { stato.alt = daCursore(+ev.target.value, ALT_MIN, ALT_MAX); aggiornaComandi(); });
el("vel").addEventListener("input", (ev) => { stato.vel = daCursore(+ev.target.value, VEL_MIN, VEL_MAX); aggiornaComandi(); });
el("lum").addEventListener("input", (ev) => { stato.lum = daCursore(+ev.target.value, LUM_MIN, LUM_MAX); aggiornaComandi(); });
el("vista1").addEventListener("click", () => impostaVista(1));
el("vista3").addEventListener("click", () => impostaVista(3));
el("persA").addEventListener("click", () => impostaPersonaggio("astronauta"));
el("persR").addEventListener("click", () => impostaPersonaggio("rover"));
el("detSi").addEventListener("click", () => { stato.dettagli = true; aggiornaComandi(); });
el("detNo").addEventListener("click", () => { stato.dettagli = false; aggiornaComandi(); });
el("ritorna").addEventListener("click", () => { stato.e = 0; stato.n = 0; stato.viaggio = null; fascio.visible = false; });
aggiornaComandi();
comandiPronti = true;

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
  stato.inMoto = false;
  let avanti = 0, lato = 0;
  if (tasti.has("KeyW") || tasti.has("ArrowUp")) avanti += 1;
  if (tasti.has("KeyS") || tasti.has("ArrowDown")) avanti -= 1;
  if (tasti.has("KeyD") || tasti.has("ArrowRight")) lato += 1;
  if (tasti.has("KeyA") || tasti.has("ArrowLeft")) lato -= 1;
  if (avanti || lato) {
    stato.viaggio = null; fascio.visible = false;          // i tasti interrompono il viaggio verso il clic
    stato.inMoto = true;
    const spinta = tasti.has("ShiftLeft") || tasti.has("ShiftRight") ? 5 : 1;
    const passo = stato.vel * spinta * dt, a = stato.dir * rad;
    const de = avanti * Math.sin(a) + lato * Math.cos(a), dn = avanti * Math.cos(a) - lato * Math.sin(a);
    const e = stato.e + passo * de, n = stato.n + passo * dn;
    stato.rotta = (Math.atan2(de, dn) / rad + 360) % 360;
    if (Math.hypot(e, n) < 0.93 * D.raggio && quota(e, n) !== null) { stato.e = e; stato.n = n; }
  } else if (stato.viaggio) {
    const v = stato.viaggio;
    stato.inMoto = true;
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
function normaleTerreno(h, p, out) {
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

  // La luce del Sole e la zona delle ombre proiettate seguono l'osservatore
  luceSole.target.position.copy(suolo);
  luceSole.position.copy(suolo).addScaledVector(versoSole, 3000);
  if (stato.dettagli) for (const S of strati) aggiornaStrato(S);

  // Personaggio: piu' scuro se si trova nell'ombra di un rilievo; passo o ruote in movimento
  const luceQui = luceSuolo(stato.e, stato.n);
  for (const m of P.materiali) {
    m.color.copy(m.userData.base).multiplyScalar(0.05 + 0.95 * luceQui);
    m.emissive.copy(m.userData.base).multiplyScalar(0.12 + 0.18 * luceQui);
  }
  if (stato.inMoto) stato.fase += dt * 9;
  const oscilla = stato.inMoto ? Math.sin(stato.fase) : 0;
  const arti = PERSONAGGI.astronauta.modello.userData.arti;
  arti.gambe.forEach((gamba, i) => { gamba.rotation.x += ((i ? -1 : 1) * 0.55 * oscilla - gamba.rotation.x) * Math.min(1, dt * 12); });
  arti.braccia.forEach((braccio, i) => { braccio.rotation.x += ((i ? 1 : -1) * 0.40 * oscilla - braccio.rotation.x) * Math.min(1, dt * 12); });
  if (stato.inMoto) for (const ruota of PERSONAGGI.rover.modello.userData.ruote) ruota.rotation.x -= dt * 7;

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
    normaleTerreno(h, P.passo, normale);
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
window.__lab = { THREE, terreni, quota, posizione3D, centroLuna, D, luceIn, strati, scena, camera, PERSONAGGI };
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
