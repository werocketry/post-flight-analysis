from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..common import feet_to_m, local_to_utc
from ..schema import finalize


def parse_blueraven_csv(path: Path, *, tz_name: str) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame()
    mapping_used: Dict[str, str] = {}

    if {"Year", "Month", "Day", "Time"}.issubset(df.columns):
        out["t_epoch_s"] = [
            local_to_utc(f"{y}-{m}-{d} {t}", tz_name)
            for y, m, d, t in zip(df["Year"], df["Month"], df["Day"], df["Time"])
        ]
    else:
        out["t_epoch_s"] = pd.NA

    if "Baro_Altitude_ASL_(feet)" in df.columns:
        out["alt_baro_m_msl"] = feet_to_m(
            pd.to_numeric(df["Baro_Altitude_ASL_(feet)"], errors="coerce")
        )
        mapping_used["Baro_Altitude_ASL_(feet)"] = "alt_baro_m_msl"
    if "Baro_Altitude_AGL_(feet)" in df.columns:
        out["alt_agl_m"] = feet_to_m(
            pd.to_numeric(df["Baro_Altitude_AGL_(feet)"], errors="coerce")
        )
        mapping_used["Baro_Altitude_AGL_(feet)"] = "alt_agl_m"

    if "Velocity_Up" in df.columns:
        out["v_up_mps"] = feet_to_m(pd.to_numeric(df["Velocity_Up"], errors="coerce"))
        mapping_used["Velocity_Up"] = "v_up_mps"
    if "Velocity_DR" in df.columns:
        out["v_dr_mps"] = feet_to_m(pd.to_numeric(df["Velocity_DR"], errors="coerce"))
        mapping_used["Velocity_DR"] = "v_dr_mps"
    if "Velocity_CR" in df.columns:
        out["v_cr_mps"] = feet_to_m(pd.to_numeric(df["Velocity_CR"], errors="coerce"))
        mapping_used["Velocity_CR"] = "v_cr_mps"

    if "Inertial_Altitude" in df.columns:
        out["alt_inertial_m"] = feet_to_m(
            pd.to_numeric(df["Inertial_Altitude"], errors="coerce")
        )
        mapping_used["Inertial_Altitude"] = "alt_inertial_m"
    if "Inertial_DR_Position" in df.columns:
        out["pos_dr_m"] = feet_to_m(
            pd.to_numeric(df["Inertial_DR_Position"], errors="coerce")
        )
        mapping_used["Inertial_DR_Position"] = "pos_dr_m"
    if "Inertial_CR_position" in df.columns:
        out["pos_cr_m"] = feet_to_m(
            pd.to_numeric(df["Inertial_CR_position"], errors="coerce")
        )
        mapping_used["Inertial_CR_position"] = "pos_cr_m"

    for vendor, std in [
        ("Tilt_Angle_(deg)", "tilt_deg"),
        ("Future_Angle_(deg)", "future_angle_deg"),
        ("Roll_Angle_(deg)", "roll_deg"),
    ]:
        if vendor in df.columns:
            out[std] = pd.to_numeric(df[vendor], errors="coerce")
            mapping_used[vendor] = std

    if "Temperature_(F)" in df.columns:
        temps_f = pd.to_numeric(df["Temperature_(F)"], errors="coerce")
        out["temperature_c"] = (temps_f - 32.0) * (5.0 / 9.0)
        mapping_used["Temperature_(F)"] = "temperature_c"

    if "Baro_Press_(atm)" in df.columns:
        atm_vals = pd.to_numeric(df["Baro_Press_(atm)"], errors="coerce")
        out["pressure_pa"] = atm_vals * 101325.0
        mapping_used["Baro_Press_(atm)"] = "pressure_pa"

    if "Batt_Volts" in df.columns:
        out["v_batt_v"] = pd.to_numeric(df["Batt_Volts"], errors="coerce")
        mapping_used["Batt_Volts"] = "v_batt_v"

    # Event/flag columns: map non‑zero values to 1, else 0
    flag_cols = {
        "Liftoff": "flag_liftoff",
        "Apogee": "flag_apogee",
        "Press_Increasing": "flag_press_increasing",
        "Burnout_Coast": "flag_burnout_coast",
        "Apo_fired": "flag_apo_fired",
        "Main_fired": "flag_main_fired",
        "3rd_fired": "flag_3rd_fired",
        "4th_fired": "flag_4th_fired",
        "Normal_Ascent": "flag_normal_ascent",
        "Accel_Vel_LE_0": "flag_accel_vel_le_0",
        "ECI_Vvel_le_0": "flag_eci_vvel_le_0",
        "Tilt Exceeded 90deg": "flag_tilt_exceeded_90deg",
    }
    for vendor, std in flag_cols.items():
        if vendor in df.columns:
            out[std] = (
                pd.to_numeric(df[vendor], errors="coerce").fillna(0) != 0
            ).astype(int)
            mapping_used[vendor] = std

    # Compute horizontal speed
    if "v_dr_mps" in out.columns and "v_cr_mps" in out.columns:
        out["speed_2d_mps"] = (
            out["v_dr_mps"].astype(float).pow(2) + out["v_cr_mps"].astype(float).pow(2)
        ).pow(0.5)

    # Compute t_flight_s anchored to first Liftoff event
    liftoff_method = "unknown"
    if "Liftoff" in df.columns:
        liftoff_num = pd.to_numeric(df["Liftoff"], errors="coerce").fillna(0)
        liftoff_mask = liftoff_num != 0
        if liftoff_mask.any():
            liftoff_iloc = int(liftoff_mask.to_numpy().argmax())
            t0_raw = out["t_epoch_s"].iloc[liftoff_iloc] if "t_epoch_s" in out.columns else None
            t0 = pd.to_numeric(pd.Series([t0_raw]), errors="coerce").iloc[0]
            if pd.notna(t0):
                t_epoch_num = pd.to_numeric(out.get("t_epoch_s", pd.Series(dtype=float)), errors="coerce")
                out["t_flight_s"] = t_epoch_num - float(t0)
                liftoff_method = "event"
    if "t_flight_s" not in out.columns:
        out["t_flight_s"] = pd.NA

    # Finalize schema
    out = finalize(out, "blueraven")

    gps_valid = int(out["lat_deg"].notna().sum()) if "lat_deg" in out.columns else 0
    row_counts = {
        "total": len(out),
        "after_clean": len(out),
        "gps_valid": gps_valid,
    }
    meta = {
        "serial": None,
        "firmware": None,
        "mapping_used": mapping_used,
        "liftoff_method": liftoff_method,
        "row_counts": row_counts,
        "units_converted": [
            "altitude_ft_to_m",
            "velocity_ft_s_to_m_s",
            "temperature_f_to_c",
            "pressure_atm_to_pa",
        ],
    }
    return out, meta
