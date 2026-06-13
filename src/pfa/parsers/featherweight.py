# src/pfa/parsers/featherweight.py
"""
Parser for FeatherWeight GPS tracker files.
Supports both CSV (text) and XLSX (spreadsheet) formats — both share the same
column layout: TRACKER, DATE, TIME, GS Lat, GS Lon, GS Alt asl,
TRACKER Lat, TRACKER Lon, TRACKER Alt asl, FIX, HORZV, VERTV, ...
Alt AGL (ft), BATT, ...

Key notes:
- All altitudes are in feet; all velocities (HORZV, VERTV) are in ft/s.
- "Alt AGL (ft)" is relative to the GROUND STATION elevation, not the launch
  site. We derive a proper AGL by subtracting the median pad altitude (rows
  where VERTV is near zero) from the GPS MSL altitude.
- DATE + TIME in CSV files are strings; in XLSX they are datetime.datetime
  and datetime.time objects (UTC as set by the user of the device).
- t_flight_s is anchored to the first row where VERTV > 20 ft/s (liftoff
  threshold), giving relative flight time independent of absolute clock.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..schema import finalize

_FT_TO_M = 0.3048
_FPS_TO_MPS = 0.3048
_LIFTOFF_VERTV_FT_S = 20.0  # ft/s threshold for liftoff detection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> pd.DataFrame:
    """Load the raw data table regardless of file format."""
    if path.suffix.lower() == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return pd.DataFrame()
        header = [str(c).strip() if c is not None else "" for c in rows[0]]
        data = [row for row in rows[1:] if any(v is not None for v in row)]
        return pd.DataFrame(data, columns=header)
    else:
        return pd.read_csv(path, skipinitialspace=True, dtype=str)


def _to_epoch(date_val, time_val) -> float | None:
    """Combine a DATE + TIME value pair into a UTC POSIX timestamp."""
    # --- date ---
    if isinstance(date_val, _dt.datetime):
        d = date_val.date()
    elif isinstance(date_val, str):
        try:
            d = _dt.date.fromisoformat(str(date_val).strip()[:10])
        except Exception:
            return None
    else:
        return None

    # --- time ---
    if isinstance(time_val, _dt.time):
        t = time_val
    elif isinstance(time_val, str):
        try:
            s = str(time_val).strip()
            parts = s.split(":")
            hh, mm = int(parts[0]), int(parts[1])
            sp = parts[2].split(".")
            ss = int(sp[0])
            us = int(sp[1].ljust(6, "0")[:6]) if len(sp) > 1 else 0
            t = _dt.time(hh, mm, ss, us)
        except Exception:
            return None
    else:
        return None

    combined = _dt.datetime.combine(d, t, tzinfo=_dt.timezone.utc)
    return combined.timestamp()


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def parse_featherweight(path: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse a FeatherWeight GPS tracker file (CSV or XLSX).
    Returns (standardised DataFrame, meta dict).
    """
    raw = _load_raw(path)
    out = pd.DataFrame()
    mapping_used: Dict[str, str] = {}

    # ---- Epoch time ----
    if "DATE" in raw.columns and "TIME" in raw.columns:
        epochs = [_to_epoch(d, t) for d, t in zip(raw["DATE"], raw["TIME"])]
        out["t_epoch_s"] = pd.to_numeric(pd.Series(epochs), errors="coerce").values
        mapping_used["DATE+TIME"] = "t_epoch_s"
    else:
        out["t_epoch_s"] = pd.NA

    # ---- GPS position ----
    for vendor, std in [("TRACKER Lat", "lat_deg"), ("TRACKER Lon", "lon_deg")]:
        if vendor in raw.columns:
            out[std] = pd.to_numeric(raw[vendor], errors="coerce")
            mapping_used[vendor] = std

    # ---- GPS altitude MSL (ft → m) ----
    if "TRACKER Alt asl" in raw.columns:
        alt_asl_ft = pd.to_numeric(raw["TRACKER Alt asl"], errors="coerce")
        out["alt_gps_m_msl"] = alt_asl_ft * _FT_TO_M
        mapping_used["TRACKER Alt asl"] = "alt_gps_m_msl"

    # ---- Velocities (ft/s → m/s) ----
    if "VERTV" in raw.columns:
        vert_ft_s = pd.to_numeric(raw["VERTV"], errors="coerce")
        out["v_up_mps"] = vert_ft_s * _FPS_TO_MPS
        mapping_used["VERTV"] = "v_up_mps"
    else:
        vert_ft_s = pd.Series(dtype=float, index=raw.index)

    if "HORZV" in raw.columns:
        out["speed_2d_mps"] = (
            pd.to_numeric(raw["HORZV"], errors="coerce").abs() * _FPS_TO_MPS
        )
        mapping_used["HORZV"] = "speed_2d_mps"

    # ---- Battery ----
    if "BATT" in raw.columns:
        out["v_batt_v"] = pd.to_numeric(raw["BATT"], errors="coerce")
        mapping_used["BATT"] = "v_batt_v"

    # ---- AGL altitude (derived from GPS MSL minus pad elevation) ----
    # "Alt AGL (ft)" in the file is relative to the ground-station elevation,
    # not the launch site, so we recompute from GPS MSL altitude.
    if "alt_gps_m_msl" in out.columns:
        on_pad = vert_ft_s.abs().fillna(999) < _LIFTOFF_VERTV_FT_S
        if on_pad.any():
            pad_alt_m = float(out["alt_gps_m_msl"][on_pad].median())
            out["alt_agl_m"] = out["alt_gps_m_msl"] - pad_alt_m
            mapping_used["TRACKER Alt asl (AGL)"] = "alt_agl_m"

    # ---- t_flight_s: anchor to first row where VERTV > threshold ----
    liftoff_method = "unknown"
    out["t_flight_s"] = pd.NA
    if "v_up_mps" in out.columns and "t_epoch_s" in out.columns:
        v_up = pd.to_numeric(out["v_up_mps"], errors="coerce")
        t_ep = pd.to_numeric(out["t_epoch_s"], errors="coerce")
        liftoff_mask = v_up > (_LIFTOFF_VERTV_FT_S * _FPS_TO_MPS)
        if liftoff_mask.any():
            t0 = float(t_ep.iloc[int(liftoff_mask.to_numpy().argmax())])
            if pd.notna(t0):
                out["t_flight_s"] = t_ep - t0
                liftoff_method = "threshold"

    # ---- Finalise ----
    out = finalize(out, "featherweight")

    gps_valid = int(out["lat_deg"].notna().sum()) if "lat_deg" in out.columns else 0
    meta = {
        "serial": None,
        "firmware": None,
        "mapping_used": mapping_used,
        "liftoff_method": liftoff_method,
        "row_counts": {
            "total": len(out),
            "after_clean": len(out),
            "gps_valid": gps_valid,
        },
        "units_converted": ["altitude_ft_to_m", "velocity_ft_s_to_m_s"],
        "notes": (
            "AGL altitude derived as GPS MSL minus median pad altitude "
            "(on-pad rows = VERTV < 20 ft/s). "
            "t_epoch_s is UTC only if the device clock was set to UTC."
        ),
    }
    return out, meta
