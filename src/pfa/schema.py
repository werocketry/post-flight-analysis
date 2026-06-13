from __future__ import annotations
import pandas as pd

STANDARD_COLUMNS = [
    "t_epoch_s",
    "t_flight_s",
    "source",
    "state",
    "event",
    "alt_baro_m_msl",
    "alt_gps_m_msl",
    "alt_agl_m",
    "alt_inertial_m",
    "lat_deg",
    "lon_deg",
    "v_up_mps",
    "v_dr_mps",
    "v_cr_mps",
    "pos_dr_m",
    "pos_cr_m",
    "speed_2d_mps",
    "a_up_mps2",
    "tilt_deg",
    "future_angle_deg",
    "roll_deg",
    "sats",
    "hdop",
    "pressure_pa",
    "temperature_c",
    "v_batt_v",
    "flag_liftoff",
    "flag_apogee",
    "flag_press_increasing",
    "flag_burnout_coast",
    "flag_apo_fired",
    "flag_main_fired",
    "flag_3rd_fired",
    "flag_4th_fired",
    "flag_normal_ascent",
    "flag_accel_vel_le_0",
    "flag_eci_vvel_le_0",
    "flag_tilt_exceeded_90deg",
    "device_sn",
    "fw_version",
    "schema_version",
]


def finalize(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    # Ensure all columns exist
    for c in STANDARD_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    # Assign constants
    df["source"] = source_name
    df["schema_version"] = "gfr-0.1"
    # Order and drop duplicates
    df = df[STANDARD_COLUMNS]
    df = df.sort_values("t_epoch_s", kind="mergesort").drop_duplicates(
        subset=["t_epoch_s"]
    )
    return df.reset_index(drop=True)
