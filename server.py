"""
Backend web di Lunar Horizon Simulator.

Serve Lunar_Horizon_Simulator_Skeleton.html e mette a disposizione della pagina
i calcoli degli script gia' presenti nella cartella, senza duplicarli:

  - terreno 3D del sito ............ horizon.py (ritaglio LOLA) + griglia di ephemeris.py
  - luce e ombra del Sole .......... ephemeris.py
  - visibilita' della Terra ........ earth_visibility.py
  - visibilita' di un satellite .... sat_visibility.py
  - punteggi landing site .......... disponibilita', blackout e sicurezza DEM
  - orizzonte 2D a 360 gradi ....... horizon.py (+ posizioni di Sole, Terra, satellite)
  - street view .................... streetview_lunare(rocks).py

Avvio:
    python server.py            # poi aprire http://127.0.0.1:5050
"""

import argparse
import base64
import json
import math
import os
import re
import subprocess
import tempfile
import sys
import threading
import time
import urllib.error
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

CARTELLA = os.path.dirname(os.path.abspath(__file__))
os.chdir(CARTELLA)  # gli script salvano lunar_dems/ e i file HTML nella cartella corrente
sys.path.insert(0, CARTELLA)

import earth_visibility  # noqa: E402
import ephemeris  # noqa: E402
import horizon  # noqa: E402
import sat_visibility  # noqa: E402

PAGINA = "Lunar_Horizon_Simulator_Skeleton.html"
PAGINA_ORIZZONTE = "horizon_view.html"
SCRIPT_STREETVIEW = "streetview_lunare(rocks).py"

LARGHEZZA_KM = 100.0  # lato della zona mostrata (come gli script)
RAGGIO_KM = 150.0  # terreno considerato per ombre e orizzonte (stesso ritaglio di horizon.py)
PUNTI = 241  # punti per lato della griglia mostrata
MAX_ISTANTI = 240  # oltre, la risposta diventa troppo pesante per il browser
ALTEZZA_OSSERVATORE_M = 2.0

COLORI = {"sun": "#ffd60a", "earth": "#4ade80", "sat": "#38bdf8"}

app = Flask(__name__, static_folder=None)


class Errore(Exception):
    """Errore da mostrare all'utente cosi' com'e'."""


@app.errorhandler(Errore)
def _errore_utente(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(Exception)
def _errore_generico(e):
    if isinstance(e, HTTPException):  # 404, 405...: restano tali
        return jsonify(error=e.description), e.code
    app.logger.exception("Errore non previsto")
    return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.after_request
def _cors(risposta):
    # Permette di usare la pagina anche aperta con un doppio clic (file://)
    risposta.headers["Access-Control-Allow-Origin"] = "*"
    risposta.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return risposta


# --------------------------------------------------------------------------
# Parametri della richiesta
# --------------------------------------------------------------------------
def leggi_sito(d):
    """Restituisce (nome, lat, lon, slug). Il nome decide il file di cache del DEM."""
    try:
        lat, lon = float(d.get("lat")), float(d.get("lon"))
    except (TypeError, ValueError):
        raise Errore("Latitudine o longitudine non valide.")
    if not (-90.0 <= lat <= 90.0) or not math.isfinite(lon):
        raise Errore("La latitudine deve essere tra -90 e 90 gradi.")
    lon = ((lon + 180.0) % 360.0) - 180.0
    for nome, lat_s, lon_s in horizon.SITI.values():
        if abs(lat - lat_s) < 1e-4 and abs(lon - (((lon_s + 180.0) % 360.0) - 180.0)) < 1e-4:
            break
    else:
        # Coordinate nel nome: due siti personalizzati diversi non condividono la cache
        nome = f"Sito {lat:+.4f} {lon:+.4f}"
    slug = "".join(c if c.isalnum() else "_" for c in nome.lower()).strip("_")
    return nome, lat, lon, slug


def leggi_tempi(d):
    """Restituisce (inizio, fine, passo) nel formato atteso da JPL Horizons."""
    def istante(chiave):
        testo = str(d.get(chiave) or "").strip().replace("T", " ")[:16]
        try:
            return datetime.strptime(testo, "%Y-%m-%d %H:%M")
        except ValueError:
            raise Errore(f"Istante '{chiave}' non valido: usare AAAA-MM-GG HH:MM (UTC).")

    inizio, fine = istante("start"), istante("end")
    if fine <= inizio:
        raise Errore("L'istante finale deve essere successivo a quello iniziale.")
    m = re.fullmatch(r"\s*(\d+)\s*([mhd])\s*", str(d.get("step") or "12 h").lower())
    if not m or int(m.group(1)) == 0:
        raise Errore("Passo non valido. Esempi: '30 m', '6 h', '1 d'.")
    minuti = int(m.group(1)) * {"m": 1, "h": 60, "d": 1440}[m.group(2)]
    n = int((fine - inizio).total_seconds() // 60 // minuti) + 1
    if n > MAX_ISTANTI:
        raise Errore(f"Troppi istanti ({n}, massimo {MAX_ISTANTI}): aumentare il passo o accorciare l'intervallo.")
    return f"{inizio:%Y-%m-%d %H:%M}", f"{fine:%Y-%m-%d %H:%M}", f"{m.group(1)} {m.group(2)}"


_cache_effemeridi = OrderedDict()
_lucchetto_horizons = threading.Lock()


def effemeridi(funzione, *argomenti):
    """
    Chiama get_lunar_ephemeris di uno degli script. Le richieste a JPL Horizons
    partono una alla volta (il servizio rifiuta con 503 quelle contemporanee),
    vengono ritentate se e' occupato e i risultati restano in memoria.
    """
    chiave = (funzione.__module__, argomenti)
    with _lucchetto_horizons:
        if chiave in _cache_effemeridi:
            _cache_effemeridi.move_to_end(chiave)
            return _cache_effemeridi[chiave]
        for tentativo in range(4):
            try:
                tabella = funzione(*argomenti)
                break
            except ValueError as e:
                # Horizons spiega il problema nel testo (oggetto sconosciuto, date fuori dall'effemeride...)
                testo = " ".join(str(e).split())
                raise Errore(f"JPL Horizons: {testo[:600]}")
            except urllib.error.HTTPError as e:
                if e.code not in (429, 502, 503) or tentativo == 3:
                    raise Errore(f"JPL Horizons ha risposto con un errore: {e}")
                time.sleep(1.5 * (tentativo + 1))
            except OSError as e:
                raise Errore(f"JPL Horizons non raggiungibile: {e}")
        _cache_effemeridi[chiave] = tabella
        while len(_cache_effemeridi) > 32:
            _cache_effemeridi.popitem(last=False)
        return tabella


# --------------------------------------------------------------------------
# Terreno e orizzonte, con cache in memoria
# --------------------------------------------------------------------------
_cache_terreno, _cache_orizzonte = OrderedDict(), OrderedDict()
_lucchetto_dem = threading.Lock()


def _ritaglio(slug, lat, lon):
    try:
        return horizon.scarica_ritaglio(horizon.USGS_LOLA_118M_URL, slug, lat, lon, RAGGIO_KM * 1000.0)
    except Exception as e:
        raise Errore(f"Scarico del DEM dal server USGS non riuscito: {e}")


def terreno(nome, lat, lon, slug):
    """Griglia locale del sito (la stessa di ephemeris.py), pronta per tutti i calcoli."""
    chiave = (round(lat, 4), round(lon, 4))
    with _lucchetto_dem:
        if chiave in _cache_terreno:
            _cache_terreno.move_to_end(chiave)
            return _cache_terreno[chiave]
        raggio_m, semilato_m = RAGGIO_KM * 1000.0, LARGHEZZA_KM * 500.0
        dem, transform, crs, _ = horizon.carica_ritaglio(_ritaglio(slug, lat, lon))
        passo_m = LARGHEZZA_KM * 1000.0 / (PUNTI - 1)
        coord, z_ext = ephemeris.griglia_locale(dem, transform, crs, lat, lon, raggio_m, passo_m)
        n_ext = (len(coord) - 1) // 2
        n_mezzo = (PUNTI - 1) // 2
        i0, i1 = n_ext - n_mezzo, n_ext + n_mezzo + 1
        z = z_ext[i0:i1, i0:i1]
        validi = ~np.isnan(z)
        if not validi.any() or not math.isfinite(float(z_ext[n_ext, n_ext])):
            raise Errore("Nel punto scelto il DEM non contiene dati.")

        dz_dn, dz_de = np.gradient(z_ext, passo_m)
        normali = np.stack([-dz_de, -dz_dn, np.ones_like(z_ext)], axis=-1)[i0:i1, i0:i1]
        normali /= np.linalg.norm(normali, axis=-1, keepdims=True)
        pendenza, roughness = metriche_terreno(z, passo_m, 0, z.shape[0])

        t = dict(
            nome=nome, lat=lat, lon=lon, z_ext=z_ext, passo_m=passo_m, n_ext=n_ext, i0=i0, i1=i1,
            z=z, validi=validi, normali=np.nan_to_num(normali), raggio_m=raggio_m,
            pendenza=pendenza, roughness=roughness,
            distanze=ephemeris.distanze_di_marcia(passo_m, math.hypot(semilato_m, semilato_m) + raggio_m),
            quota_centro_m=float(z_ext[n_ext, n_ext]),
        )
        _cache_terreno[chiave] = t
        while len(_cache_terreno) > 4:
            _cache_terreno.popitem(last=False)
        return t


def orizzonte(lat, lon, slug):
    chiave = (round(lat, 4), round(lon, 4))
    with _lucchetto_dem:
        if chiave not in _cache_orizzonte:
            dem, transform, crs, passo = horizon.carica_ritaglio(_ritaglio(slug, lat, lon))
            _cache_orizzonte[chiave] = horizon.calcola_orizzonte(
                dem, transform, crs, lat, lon, ALTEZZA_OSSERVATORE_M, RAGGIO_KM * 1000.0, 0.5, passo_d=passo
            )
            while len(_cache_orizzonte) > 6:
                _cache_orizzonte.popitem(last=False)
        return _cache_orizzonte[chiave]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def metriche_terreno(z_ext, passo_m, i0, i1):
    """Return local slope in degrees and detrended 3x3 roughness in metres."""
    dz_dn, dz_de = np.gradient(z_ext, passo_m)
    pendenza = np.degrees(np.arctan(np.hypot(dz_de, dz_dn)))[i0:i1, i0:i1]

    z = z_ext[i0:i1, i0:i1]
    righe, colonne = z.shape
    validi = np.isfinite(z)
    conteggio = np.zeros(z.shape, dtype=np.float64)
    somma_x = np.zeros(z.shape, dtype=np.float64)
    somma_y = np.zeros(z.shape, dtype=np.float64)
    somma_z = np.zeros(z.shape, dtype=np.float64)
    somma_xx = np.zeros(z.shape, dtype=np.float64)
    somma_xy = np.zeros(z.shape, dtype=np.float64)
    somma_yy = np.zeros(z.shape, dtype=np.float64)
    somma_xz = np.zeros(z.shape, dtype=np.float64)
    somma_yz = np.zeros(z.shape, dtype=np.float64)
    somma_zz = np.zeros(z.shape, dtype=np.float64)

    for dy in range(-1, 2):
        centro_r = slice(max(0, -dy), min(righe, righe - dy))
        vicino_r = slice(max(0, dy), min(righe, righe + dy))
        for dx in range(-1, 2):
            centro_c = slice(max(0, -dx), min(colonne, colonne - dx))
            vicino_c = slice(max(0, dx), min(colonne, colonne + dx))
            quota = z[vicino_r, vicino_c]
            valido = np.isfinite(quota)
            x, y = float(dx), float(dy)
            conteggio[centro_r, centro_c] += valido
            somma_x[centro_r, centro_c] += np.where(valido, x, 0.0)
            somma_y[centro_r, centro_c] += np.where(valido, y, 0.0)
            somma_z[centro_r, centro_c] += np.where(valido, quota, 0.0)
            somma_xx[centro_r, centro_c] += np.where(valido, x * x, 0.0)
            somma_xy[centro_r, centro_c] += np.where(valido, x * y, 0.0)
            somma_yy[centro_r, centro_c] += np.where(valido, y * y, 0.0)
            somma_xz[centro_r, centro_c] += np.where(valido, x * quota, 0.0)
            somma_yz[centro_r, centro_c] += np.where(valido, y * quota, 0.0)
            somma_zz[centro_r, centro_c] += np.where(valido, quota * quota, 0.0)

    matrice = np.empty(z.shape + (3, 3), dtype=np.float64)
    matrice[..., 0, 0], matrice[..., 0, 1], matrice[..., 0, 2] = somma_xx, somma_xy, somma_x
    matrice[..., 1, 0], matrice[..., 1, 1], matrice[..., 1, 2] = somma_xy, somma_yy, somma_y
    matrice[..., 2, 0], matrice[..., 2, 1], matrice[..., 2, 2] = somma_x, somma_y, conteggio
    termine = np.stack((somma_xz, somma_yz, somma_z), axis=-1)
    roughness = np.full(z.shape, np.nan, dtype=np.float64)
    determinante = np.linalg.det(matrice)
    stimabili = validi & (conteggio >= 6) & np.isfinite(determinante) & (np.abs(determinante) > 1e-10)
    coefficienti = np.linalg.solve(matrice[stimabili], termine[stimabili, :, None])[..., 0]
    errore_quadratico = somma_zz[stimabili] - np.sum(coefficienti * termine[stimabili], axis=-1)
    roughness[stimabili] = np.sqrt(np.maximum(errore_quadratico / conteggio[stimabili], 0.0))

    pendenza[~validi] = np.nan
    return pendenza.astype("<f4"), roughness.astype("<f4")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.post("/api/terrain")
def api_terreno():
    """Terreno 3D del sito (vista di lunar.py), sulla griglia usata dagli altri calcoli."""
    t = terreno(*leggi_sito(request.get_json(force=True)))
    quota_min = float(np.nanmin(t["z"]))
    z_km = ((t["z"] - quota_min) / 1000.0).astype("<f4")  # NaN = senza dato
    return jsonify(
        name=t["nome"], lat=t["lat"], lon=t["lon"], n=PUNTI, side_km=LARGHEZZA_KM,
        relief_km=float(np.nanmax(z_km)), center_z_km=(t["quota_centro_m"] - quota_min) / 1000.0,
        center_elev_m=t["quota_centro_m"], z=b64(z_km),
        slope=b64(t["pendenza"]), roughness=b64(t["roughness"]), grid_spacing_m=t["passo_m"],
    )


def prepara_strato(tipo, d, t, inizio, fine, passo):
    """Prepare a visibility time series shared by the map and score endpoints."""
    lat, lon = t["lat"], t["lon"]
    comuni = (t["z_ext"], t["passo_m"], t["n_ext"], t["i0"], t["i1"])
    extra = {}

    if tipo == "sun":
        tabella = effemeridi(ephemeris.get_lunar_ephemeris, "10", lat, lon, 0, inizio, fine, passo)
        elevazione_buio = -math.degrees(t["raggio_m"] / horizon.R_LUNA) - 2.0 * ephemeris.RAGGIO_SOLE_GRADI

        def fotogramma(az, el):
            if el < elevazione_buio:
                return np.zeros(t["validi"].shape)
            return ephemeris.illuminazione(*comuni, t["normali"], az, el, t["distanze"])

    elif tipo == "earth":
        tabella = effemeridi(earth_visibility.get_lunar_ephemeris, "399", lat, lon, 0, inizio, fine, passo)

        def fotogramma(az, el):
            return earth_visibility.visibilita_terra(*comuni, az, el, t["distanze"])

    elif tipo == "sat":
        satellite = str(d.get("satellite") or "").strip().strip("\"'")
        if not satellite:
            raise Errore("Inserire il nome o l'ID Horizons del satellite (es. -85 per LRO).")
        tabella = effemeridi(
            sat_visibility.get_lunar_ephemeris, satellite, lat, lon,
            t["quota_centro_m"] / 1000.0, inizio, fine, passo
        )
        extra = dict(satellite=satellite, range_km=[r[3] for r in tabella])

        def fotogramma(az, el, distanza_km):
            if el < -20.0:
                return np.zeros(t["validi"].shape)
            return sat_visibility.visibilita_satellite(
                *comuni, az, el, distanza_km, t["quota_centro_m"], t["distanze"]
            )
    else:
        raise Errore("Strato sconosciuto.")

    return tabella, fotogramma, extra


@app.post("/api/layer")
def api_strato():
    """Uno strato animato: 'sun' (luce e ombra), 'earth' o 'sat' (visibilita')."""
    d = request.get_json(force=True)
    tipo = d.get("kind")
    inizio, fine, passo = leggi_tempi(d)
    t = terreno(*leggi_sito(d))
    validi = t["validi"]
    tabella, fotogramma, extra = prepara_strato(tipo, d, t, inizio, fine, passo)

    colori, percentuali = [], []
    for riga in tabella:
        valori = fotogramma(*riga[1:])
        valori[~validi] = 0.0
        if tipo == "sun":
            colori.append(ephemeris.in_scala_colori(valori))
            visibili = valori > 1e-3
        elif tipo == "earth":
            colori.append(earth_visibility.in_scala_colori(valori))
            visibili = valori > 1e-3
        else:
            colori.append(sat_visibility.in_scala_colori(valori))
            visibili = valori > 0.5
        percentuali.append(round(100.0 * float(np.mean(visibili[validi])), 2))
    return jsonify(
        kind=tipo, n=PUNTI, times=[f"{r[0]:%Y-%m-%d %H:%M}" for r in tabella],
        az=[r[1] for r in tabella], el=[r[2] for r in tabella], pct=percentuali,
        data=b64(np.stack(colori)), **extra,
    )


@app.post("/api/scores")
def api_punteggi():
    """Compute temporal visibility and terrain-safety scores without sending map frames."""
    d = request.get_json(force=True)
    inizio, fine, passo = leggi_tempi(d)
    t = terreno(*leggi_sito(d))
    satellite = str(d.get("satellite") or "").strip().strip("\"'")
    selezioni = ["sun", "earth"] + (["sat"] if satellite else [])
    serie = []
    for kind in selezioni:
        tabella, fotogramma, _ = prepara_strato(kind, d, t, inizio, fine, passo)
        if serie and [r[0] for r in tabella] != [r[0] for r in serie[0]["tabella"]]:
            raise Errore("Le effemeridi non condividono la stessa griglia temporale.")
        serie.append(dict(kind=kind, tabella=tabella, fotogramma=fotogramma))

    times = serie[0]["tabella"]
    timestamps = np.array([r[0].timestamp() for r in times], dtype=np.float64)
    inizio_utc = datetime.strptime(inizio, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
    fine_utc = datetime.strptime(fine, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
    durata_finestra = fine_utc - inizio_utc
    durate = np.array([
        max(0.0, min(timestamps[k + 1] if k + 1 < len(times) else fine_utc, fine_utc)
            - max(istante, inizio_utc))
        for k, istante in enumerate(timestamps)
    ])
    durata_analizzata = float(durate.sum())
    if durata_finestra <= 0 or durata_analizzata <= 0:
        raise Errore("La finestra selezionata non contiene intervalli temporali analizzabili.")

    validi = t["validi"]
    valid_count = int(np.count_nonzero(validi))
    accumuli = {}
    for item in serie:
        accumuli[item["kind"]] = dict(
            visible_seconds=0.0,
            current_blackout=np.zeros(validi.shape, dtype=np.float64),
            longest_blackout=np.zeros(validi.shape, dtype=np.float64),
            current_visible=np.zeros(validi.shape, dtype=np.float64),
            longest_visible=np.zeros(validi.shape, dtype=np.float64),
        )
    contemporaneo = dict(
        visible_seconds=0.0,
        current_blackout=np.zeros(validi.shape, dtype=np.float64),
        longest_blackout=np.zeros(validi.shape, dtype=np.float64),
        current_visible=np.zeros(validi.shape, dtype=np.float64),
        longest_visible=np.zeros(validi.shape, dtype=np.float64),
    )

    for index, _ in enumerate(times):
        maschere = {}
        for item in serie:
            riga = item["tabella"][index]
            valori = item["fotogramma"](*riga[1:])
            soglia = 1e-3 if item["kind"] in ("sun", "earth") else 0.5
            maschere[item["kind"]] = (valori > soglia) & validi
        insieme = np.logical_and.reduce(list(maschere.values()))
        durata = durate[index]

        for kind, visibile in [*maschere.items(), ("simultaneous", insieme)]:
            stato = contemporaneo if kind == "simultaneous" else accumuli[kind]
            stato["visible_seconds"] += float(np.count_nonzero(visibile)) * durata
            blackout = np.where(validi & ~visibile, stato["current_blackout"] + durata, 0.0)
            visibilita = np.where(validi & visibile, stato["current_visible"] + durata, 0.0)
            np.maximum(stato["longest_blackout"], blackout, out=stato["longest_blackout"])
            np.maximum(stato["longest_visible"], visibilita, out=stato["longest_visible"])
            stato["current_blackout"] = blackout
            stato["current_visible"] = visibilita

    def riepilogo(stato):
        disponibilita = 100.0 * stato["visible_seconds"] / (valid_count * durata_analizzata)
        blackout_medio = float(np.sum(stato["longest_blackout"][validi])) / valid_count
        periodo_medio = float(np.sum(stato["longest_visible"][validi])) / valid_count
        continuita = 100.0 * (1.0 - min(blackout_medio / durata_finestra, 1.0))
        return dict(
            availability_pct=round(disponibilita, 2),
            mean_max_blackout_h=round(blackout_medio / 3600.0, 2),
            mean_longest_visibility_h=round(periodo_medio / 3600.0, 2),
            score=round(0.75 * disponibilita + 0.25 * continuita, 2),
        )

    try:
        max_slope = float(d.get("max_slope", 15.0))
        max_roughness = float(d.get("max_roughness", 20.0))
    except (TypeError, ValueError):
        raise Errore("Le soglie di pendenza e rugosita' devono essere numeriche.")
    if not math.isfinite(max_slope) or max_slope < 0 or max_slope > 90:
        raise Errore("La pendenza massima deve essere compresa tra 0 e 90 gradi.")
    if not math.isfinite(max_roughness) or max_roughness < 0:
        raise Errore("La rugosita' massima deve essere un valore non negativo.")

    terrain_valid = validi & np.isfinite(t["pendenza"]) & np.isfinite(t["roughness"])
    safety_count = int(np.count_nonzero(terrain_valid))
    if not safety_count:
        raise Errore("Il DEM non contiene celle valide per stimare la sicurezza del terreno.")
    safety_pct = 100.0 * float(np.count_nonzero(
        terrain_valid & (t["pendenza"] <= max_slope) & (t["roughness"] <= max_roughness)
    )) / safety_count
    return jsonify(
        site=t["nome"], satellite=satellite or None, sample_step=passo,
        window_hours=round(durata_finestra / 3600.0, 2), analyzed_hours=round(durata_analizzata / 3600.0, 2),
        valid_cells=valid_count,
        visibility={kind: riepilogo(accumuli[kind]) for kind in selezioni},
        simultaneous=riepilogo(contemporaneo),
        terrain_safety=dict(
            score=round(safety_pct, 2), max_slope=max_slope, max_roughness=max_roughness,
            valid_cells=safety_count, grid_spacing_m=round(t["passo_m"], 2),
        ),
    )


def _traccia(fig, nome, colore, simbolo, tabella, az_or, el_or):
    """Aggiunge al panorama il percorso di un corpo nell'intervallo scelto."""
    import plotly.graph_objects as go

    az = np.array([r[1] % 360.0 for r in tabella])
    el = np.array([r[2] for r in tabella])
    sopra = el > np.interp(az, az_or, el_or, period=360.0)
    testo = [f"{r[0]:%Y-%m-%d %H:%M} UTC<br>" + ("sopra" if s else "sotto") + " l'orizzonte locale"
             for r, s in zip(tabella, sopra)]
    # Interrompe la linea quando il corpo attraversa il nord (359 -> 0 gradi)
    x, y, t = [], [], []
    for k in range(len(az)):
        if k and abs(az[k] - az[k - 1]) > 180.0:
            x.append(None), y.append(None), t.append("")
        x.append(float(az[k])), y.append(float(el[k])), t.append(testo[k])
    for s in (-360.0, 0.0, 360.0):  # stesse tre copie affiancate del panorama
        fig.add_trace(go.Scatter(
            x=[None if v is None else v + s for v in x], y=y, text=t, mode="lines+markers",
            name=nome, legendgroup=nome, showlegend=(s == 0.0),
            line=dict(color=colore, width=1.2, dash="dot"),
            marker=dict(size=7, color=colore, symbol=simbolo, line=dict(color="#05060a", width=0.5)),
            hovertemplate=f"<b>{nome}</b><br>%{{text}}<br>Azimut %{{x:.1f}}°<br>Elevazione %{{y:.2f}}°<extra></extra>",
        ))
    # Posizione all'istante iniziale, ben visibile
    for s in (-360.0, 0.0, 360.0):
        fig.add_trace(go.Scatter(
            x=[float(az[0]) + s], y=[float(el[0])], mode="markers+text", showlegend=False, legendgroup=nome,
            marker=dict(size=15, color=colore, symbol=simbolo, line=dict(color="white", width=1.2)),
            text=[nome], textposition="top center", textfont=dict(color=colore, size=12), hoverinfo="skip",
        ))
    return dict(name=nome, visible=int(sopra.sum()), total=len(tabella),
                el_min=float(el.min()), el_max=float(el.max()))


@app.post("/api/horizon")
def api_orizzonte():
    """Panorama 2D di horizon.py con le posizioni di Sole, Terra ed eventuale satellite."""
    d = request.get_json(force=True)
    nome, lat, lon, slug = leggi_sito(d)
    inizio, fine, passo = leggi_tempi(d)
    az, piani, dist_km, quota0 = orizzonte(lat, lon, slug)
    el_or = piani[-1][1]
    fig = horizon.disegna(nome, az, piani, dist_km, lat, lon, ALTEZZA_OSSERVATORE_M, quota0, os.devnull)

    corpi, avvisi = [], []
    richieste = [
        ("Sole", COLORI["sun"], "circle", ephemeris.get_lunar_ephemeris, "10"),
        ("Terra", COLORI["earth"], "circle", earth_visibility.get_lunar_ephemeris, "399"),
    ]
    satellite = str(d.get("satellite") or "").strip().strip("\"'")
    if satellite:
        richieste.append((satellite, COLORI["sat"], "diamond", sat_visibility.get_lunar_ephemeris, satellite))
    for etichetta, colore, simbolo, funzione, bersaglio in richieste:
        try:
            tabella = effemeridi(funzione, bersaglio, lat, lon, quota0 / 1000.0, inizio, fine, passo)
        except Errore as e:
            if bersaglio in ("10", "399"):
                raise
            avvisi.append(f"{etichetta}: {e}")  # il satellite e' facoltativo: il resto si mostra comunque
            continue
        corpi.append(_traccia(fig, etichetta, colore, simbolo, tabella, az, el_or))

    # L'asse verticale deve contenere anche i corpi alti nel cielo
    y0, y1 = fig.layout.yaxis.range
    y1 = max(y1, min(90.0, max(c["el_max"] for c in corpi) + 4.0))
    fig.update_layout(yaxis=dict(range=[y0, y1], fixedrange=False), legend=dict(orientation="h", y=-0.15))
    info = dict(bodies=corpi, warnings=avvisi, window=f"{inizio} → {fine} UTC, every {passo}")
    corpo = '{"fig":' + fig.to_json() + ',"info":' + json.dumps(info) + "}"
    return Response(corpo, mimetype="application/json")


@app.post("/api/streetview")
def api_streetview():
    """Genera la pagina di streetview_lunare(rocks).py per il sito e restituisce il suo indirizzo."""
    d = request.get_json(force=True)
    nome, lat, lon, slug = leggi_sito(d)
    inizio, fine, passo = leggi_tempi(d)
    # Sole e Terra: le stesse identiche tabelle (e la stessa cache) degli strati di luce e di
    # visibilita' della Terra, cosi' la vista dal suolo mostra le posizioni calcolate dagli altri script
    sole = effemeridi(ephemeris.get_lunar_ephemeris, "10", lat, lon, 0, inizio, fine, passo)
    terra = effemeridi(earth_visibility.get_lunar_ephemeris, "399", lat, lon, 0, inizio, fine, passo)
    if [r[0] for r in sole] != [r[0] for r in terra]:
        raise Errore("Le effemeridi non condividono la stessa griglia temporale.")
    cielo = [[f"{s[0]:%Y-%m-%d %H:%M} UTC", s[1], s[2], e[1], e[2]] for s, e in zip(sole, terra)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(cielo, f)
    try:
        comando = [
            sys.executable, SCRIPT_STREETVIEW, "--nome", nome, f"--lat={lat}", f"--lon={lon}",
            "--istante", "", "--cielo", f.name, "--no-browser",
        ]
        esito = subprocess.run(comando, cwd=CARTELLA, capture_output=True, text=True, timeout=900)
    finally:
        os.unlink(f.name)
    if esito.returncode != 0:
        righe = (esito.stderr or esito.stdout).strip().splitlines()
        raise Errore("Street view non generato: " + (righe[-1] if righe else "errore sconosciuto"))
    # Istante della linea del tempo del simulatore: la vista si apre su quello
    try:
        k = min(max(int(d.get("k")), 0), len(cielo) - 1)
    except (TypeError, ValueError):
        k = None
    return jsonify(url=f"/streetview/file/{slug}" + (f"?k={k}" if k is not None else ""))


@app.get("/streetview/file/<slug>")
def file_streetview(slug):
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise Errore("Sito non valido.")
    percorso = os.path.join(CARTELLA, f"streetview_{slug}.html")
    if not os.path.isfile(percorso):
        raise Errore("Street view non ancora generato per questo sito.")
    return send_file(percorso, max_age=0)


# --------------------------------------------------------------------------
# Pagine
# --------------------------------------------------------------------------
PAGINA_ATTESA_STREETVIEW = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Lunar Street View</title>
<style>body{margin:0;height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;
gap:14px;background:#020617;color:#cbd5e1;font:14px ui-monospace,Menlo,monospace;text-align:center;padding:0 24px}
.s{width:34px;height:34px;border:3px solid #334155;border-top-color:#818cf8;border-radius:50%;animation:r 1s linear infinite}
@keyframes r{to{transform:rotate(360deg)}} .e{color:#fb7185}</style></head>
<body><div class="s" id="s"></div><div id="m">Generating the street view (terrain + Sun/Earth ephemeris)…</div>
<script>
const p = Object.fromEntries(new URLSearchParams(location.search));
fetch('/api/streetview', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(p)})
  .then(r => r.json()).then(j => { if (j.error) throw new Error(j.error); location.replace(j.url); })
  .catch(e => { document.getElementById('s').remove(); const m = document.getElementById('m'); m.className = 'e'; m.textContent = e.message; });
</script></body></html>"""


@app.get("/")
def pagina():
    return send_file(os.path.join(CARTELLA, PAGINA), max_age=0)


@app.get("/horizon")
def pagina_orizzonte():
    return send_file(os.path.join(CARTELLA, PAGINA_ORIZZONTE), max_age=0)


@app.get("/streetview")
def pagina_streetview():
    return Response(PAGINA_ATTESA_STREETVIEW, mimetype="text/html")


@app.get("/plotly.min.js")
def plotly_js():
    # La stessa versione di plotly.js con cui Python ha costruito le figure
    import plotly

    return send_file(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"),
                     mimetype="application/javascript", max_age=86400)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interfaccia web di Lunar Horizon Simulator.")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--no-browser", action="store_true", help="non aprire il browser all'avvio")
    a = p.parse_args()
    indirizzo = f"http://127.0.0.1:{a.port}"
    print(f"Lunar Horizon Simulator: {indirizzo}")
    if not a.no_browser:
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=[indirizzo]).start()
    app.run(host="127.0.0.1", port=a.port, threaded=True)
