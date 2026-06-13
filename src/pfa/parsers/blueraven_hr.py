# src/pfa/parsers/blueraven_hr.py
"""
Parser for Blue Raven high-rate (HR) IMU CSV files.

The HR file records at ~500 Hz and contains only inertial data:
  Year, Month, Day, Time, Flight_Time_(s), Sync,
  Gyro_X, Gyro_Y, Gyro_Z,          [deg/s]
  Accel_X, Accel_Y, Accel_Z,        [g]
  Quat_1, Quat_2, Quat_3, Quat_4,  [dimensionless]
  Aux_Volts, Current

No barometric, GPS, or velocity data is present.

Axis convention (empirically observed):
  Accel_Y ≈ -1 g at rest → Y is the rocket axial (nose-up) axis in body frame,
  reading negative specific force in that axis at rest due to device convention.
  Accel_Y rises to ~+26 g during motor burn.
  Gyro values are in deg/s.

The standard `a_up_mps2` column is populated as Accel_Y * 9.80665 so that the
HR data can appear in the acceleration panel of flight-profile plots.
The raw g-unit columns (accel_x_g, accel_y_g, accel_z_g, gyro_x_degps, ...,
quat_1..4) are also stored via the extended schema.

Flight_Time_(s) is already relative to liftoff (negative = pre-launch, 0 = liftoff)
and is used directly as t_flight_s.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..common import local_to_utc
from ..schema import finalize

_G = 9.80665


def parse_blueraven_hr_csv(path: Path, *, tz_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse a Blue Raven high-rate IMU CSV file.
    Returns (standardised DataFrame, meta dict).
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame()
    mapping_used: Dict[str, str] = {}

    # ---- Epoch time (same construction as Blue Raven LR) ----
    if {"Year", "Month", "Day", "Time"}.issubset(df.columns):
        out["t_epoch_s"] = [
            local_to_utc(f"{y}-{m}-{d} {t}", tz_name)
            for y, m, d, t in zip(df["Year"], df["Month"], df["Day"], df["Time"])
        ]
    else:
        out["t_epoch_s"] = pd.NA

    # ---- Flight time: use device-reported value directly ----
    if "Flight_Time_(s)" in df.columns:
        out["t_flight_s"] = pd.to_numeric(df["Flight_Time_(s)"], errors="coerce")
        mapping_used["Flight_Time_(s)"] = "t_flight_s"
    else:
        out["t_flight_s"] = pd.NA

    # ---- Accelerometers (g) ----
    for vendor, std in [
        ("Accel_X", "accel_x_g"),
        ("Accel_Y", "accel_y_g"),
        ("Accel_Z", "accel_z_g"),
    ]:
        if vendor in df.columns:
            out[std] = pd.to_numeric(df[vendor], errors="coerce")
            mapping_used[vendor] = std

    # Axial upward acceleration in m/s²: Accel_Y is the nose-up axis
    if "accel_y_g" in out.columns:
        out["a_up_mps2"] = out["accel_y_g"] * _G
        mapping_used["Accel_Y (axial)"] = "a_up_mps2"

    # ---- Gyroscopes (deg/s) ----
    for vendor, std in [
        ("Gyro_X", "gyro_x_degps"),
        ("Gyro_Y", "gyro_y_degps"),
        ("Gyro_Z", "gyro_z_degps"),
    ]:
        if vendor in df.columns:
            out[std] = pd.to_numeric(df[vendor], errors="coerce")
            mapping_used[vendor] = std

    # ---- Quaternion (dimensionless) ----
    for vendor, std in [
        ("Quat_1", "quat_1"),
        ("Quat_2", "quat_2"),
        ("Quat_3", "quat_3"),
        ("Quat_4", "quat_4"),
    ]:
        if vendor in df.columns:
            out[std] = pd.to_numeric(df[vendor], errors="coerce")
            mapping_used[vendor] = std

    # ---- Finalise ----
    out = finalize(out, "blueraven_hr")

    meta = {
        "serial": None,
        "firmware": None,
        "mapping_used": mapping_used,
        "liftoff_method": "device_reported",
        "row_counts": {
            "total": len(out),
            "after_clean": len(out),
            "gps_valid": 0,
        },
        "units_converted": [],
        "notes": (
            "a_up_mps2 = Accel_Y * g (Accel_Y is the rocket axial/nose-up axis). "
            "Gyro values assumed to be in deg/s. "
            "Quaternion components are dimensionless."
        ),
    }
    return out, meta
