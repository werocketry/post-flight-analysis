from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..schema import finalize

COLUMN_MAP = {
    "state_name": "state",
    "acceleration": "a_up_mps2",
    "pressure": "pressure_pa",
    "altitude": "alt_baro_m_msl",
    "height": None,
    "speed": "v_up_mps",
    "temperature": "temperature_c",
    "battery_voltage": "v_batt_v",
    "batt_voltage": "v_batt_v",
    "nsat": "sats",
    "latitude": "lat_deg",
    "longitude": "lon_deg",
    "hdop": "hdop",
    "year": None,
    "month": None,
    "day": None,
    "hour": None,
    "minute": None,
    "second": None,
}


def _compose_epoch_seconds(row: pd.Series) -> float:
    """Use discrete Y/M/D/H/M/S and fractional 'time' to build epoch seconds."""
    from datetime import datetime, timezone, timedelta

    dt = datetime(
        int(row["year"]),
        int(row["month"]),
        int(row["day"]),
        int(row["hour"]),
        int(row["minute"]),
        int(row["second"]),
        tzinfo=timezone.utc,
    )
    if "time" in row and pd.notna(row["time"]):
        try:
            offset = float(row["time"])
        except Exception:
            offset = 0.0
        dt = dt + timedelta(seconds=offset)
    return dt.timestamp()


def _infer_t_flight(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Liftoff from state if available, else acceleration threshold."""
    liftoff_idx = None
    if "state" in df.columns:
        state_series = (
            df.get("state", pd.Series(index=df.index, dtype=str))
            .astype(str)
            .str.lower()
        )
        matches = state_series.isin(["boost", "fast"])
        if matches.any():
            liftoff_idx = df.index[matches].min()

    method = (
        "event" if liftoff_idx is not None and pd.notna(liftoff_idx) else "threshold"
    )

    if liftoff_idx is None or pd.isna(liftoff_idx):
        accel_series = df.get("a_up_mps2", pd.Series(index=df.index, dtype=float))
        accel_num = pd.to_numeric(accel_series, errors="coerce")
        if (accel_num > 30).any():
            liftoff_idx = df.index[(accel_num > 30)].min()

    if liftoff_idx is None or pd.isna(liftoff_idx):
        return pd.Series([pd.NA] * len(df), index=df.index), "unknown"

    t0_raw = df.loc[liftoff_idx, "t_epoch_s"]
    t0 = pd.to_numeric(pd.Series([t0_raw]), errors="coerce").iloc[0]
    if pd.isna(t0):
        return pd.Series([pd.NA] * len(df), index=df.index), "unknown"

    t_series = pd.to_numeric(
        df.get("t_epoch_s", pd.Series(index=df.index, dtype=float)), errors="coerce"
    )
    return t_series - float(t0), method


def parse_telemega_csv(path: Path) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(
        path,
        header=0,
        skipinitialspace=True,
        engine="python",
    )
    cols = [c.strip() for c in df.columns]
    if cols:
        cols[0] = cols[0].lstrip("#\ufeff").strip()
    df.columns = cols

    out = pd.DataFrame()
    mapping_used: Dict[str, str] = {}

    required_time = {"year", "month", "day", "hour", "minute", "second"}
    if required_time.issubset(set(df.columns)):
        out["t_epoch_s"] = df.apply(_compose_epoch_seconds, axis=1)
    else:
        out["t_epoch_s"] = pd.NA

    for vendor, std in COLUMN_MAP.items():
        if std is None or vendor not in df.columns:
            continue
        if std == "state":
            out[std] = df[vendor].astype(str)
        else:
            out[std] = pd.to_numeric(df[vendor], errors="coerce")
        mapping_used[vendor] = std

    if "height" in df.columns:
        out["alt_agl_m"] = pd.to_numeric(df["height"], errors="coerce")
        mapping_used["height"] = "alt_agl_m"

    gps_alt_cols = [
        c
        for c in list(df.columns)
        if re.fullmatch(r"altitude(\.\d+)?", str(c)) and str(c) != "altitude"
    ]
    if gps_alt_cols:
        out["alt_gps_m_msl"] = pd.to_numeric(df[gps_alt_cols[-1]], errors="coerce")
        mapping_used[gps_alt_cols[-1]] = "alt_gps_m_msl"

    if "a_up_mps2" not in out.columns and "accel_z" in df.columns:
        out["a_up_mps2"] = pd.to_numeric(df["accel_z"], errors="coerce")
        mapping_used["accel_z"] = "a_up_mps2"

    t_flight, method = _infer_t_flight(out)
    out["t_flight_s"] = t_flight

    serial = str(df["serial"].iloc[0]) if "serial" in df.columns else None
    fw = str(df["version"].iloc[0]) if "version" in df.columns else None
    out["device_sn"] = serial
    out["fw_version"] = fw

    out = finalize(out, "telemega")

    gps_valid = int(out["lat_deg"].notna().sum()) if "lat_deg" in out.columns else 0
    row_counts = {
        "total": len(out),
        "after_clean": len(out),
        "gps_valid": gps_valid,
    }
    meta = {
        "serial": serial,
        "firmware": fw,
        "mapping_used": mapping_used,
        "liftoff_method": method,
        "row_counts": row_counts,
        "units_converted": [],
    }
    return out, meta
