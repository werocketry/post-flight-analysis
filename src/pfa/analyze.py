# src/pfa/analyze.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .common import ensure_dir

_M_TO_FT = 3.28084
_MPS_TO_FPS = 3.28084
_G = 9.80665

# Colours and labels for each data source
_SOURCE_COLOR = {
    "telemega_std":    "#1f77b4",
    "blueraven_std":   "#ff7f0e",
    "featherweight_std": "#2ca02c",
    "blueraven_hr_std": "#9467bd",
}
_SOURCE_LABEL = {
    "telemega_std":    "TeleMega",
    "blueraven_std":   "Blue Raven",
    "featherweight_std": "FeatherWeight GPS",
    "blueraven_hr_std": "Blue Raven HR (IMU)",
}
# Sources that carry meaningful alt/vel data for the main plot panels
_PLOT_SOURCES = {"telemega_std", "blueraven_std", "featherweight_std"}

# TeleMega state names to watch for event transitions
_TM_EVENT_STATES = [
    ("coast", "Burnout"),   # first "coast" or "fast" row marks motor burnout
    ("fast",  "Burnout"),
    ("drogue", "Drogue"),
    ("main",   "Main"),
    ("landed", "Landed"),
]
# BlueRaven flag columns → event label (only if not already set by TeleMega)
_BR_FLAG_EVENTS = [
    ("flag_burnout_coast", "Burnout"),
    ("flag_apogee",        "Apogee"),
    ("flag_apo_fired",     "Drogue"),
    ("flag_main_fired",    "Main"),
]
_EVENT_COLOR = {
    "Burnout": "#d62728",
    "Apogee":  "#2ca02c",
    "Drogue":  "#9467bd",
    "Main":    "#8c564b",
    "Landed":  "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_interim(interim: Path) -> Dict[str, pd.DataFrame]:
    """Load every *_std.csv from interim/ keyed by stem."""
    datasets: Dict[str, pd.DataFrame] = {}
    for p in sorted(interim.glob("*_std.csv")):
        datasets[p.stem] = pd.read_csv(p)
    return datasets


# Expose as public name for CLI compare command
load_interim = _load_interim


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series for col, all-NaN if col absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(dtype=float, index=df.index)


def _velocity_for_plot(df: pd.DataFrame) -> pd.Series:
    """
    Best-available vertical velocity for plotting.
    Prefers v_up_mps when it is dense and large (Blue Raven Kalman output).
    Falls back to a smoothed baro-altitude derivative for TeleMega, where
    v_up_mps is GPS-derived and mostly absent during the boost phase.
    """
    vel = _num(df, "v_up_mps")
    t   = _num(df, "t_flight_s")
    in_flight = t >= 0

    vel_in_flight = vel[in_flight]
    if (vel_in_flight.notna().mean() > 0.5
            and float(vel_in_flight.max()) > 50.0):
        return vel

    # Smoothed finite-difference of baro altitude
    alt = _num(df, "alt_agl_m")
    dt  = t.diff()
    dh  = alt.diff()
    v_baro = (dh / dt).where((dt > 0.001) & (dt < 0.5))
    return v_baro.rolling(31, center=True, min_periods=10).mean()


def _detect_events(df: pd.DataFrame) -> Dict[str, float]:
    """
    Return {event_label: t_flight_s} for key flight events.
    Works with TeleMega state column or BlueRaven flag columns.
    Duplicate labels are ignored (first found wins).
    """
    events: Dict[str, float] = {}
    t = _num(df, "t_flight_s")
    in_flight = t >= 0

    # TeleMega state-based events
    if "state" in df.columns:
        state_s = df["state"].astype(str).str.lower().str.strip()
        for state_name, label in _TM_EVENT_STATES:
            if label in events:
                continue
            mask = (state_s == state_name) & in_flight
            if mask.any():
                events[label] = float(t[mask].min())

    # BlueRaven flag-based events (fill gaps only)
    for flag_col, label in _BR_FLAG_EVENTS:
        if label in events or flag_col not in df.columns:
            continue
        flag_num = _num(df, flag_col).fillna(0)
        mask = (flag_num != 0) & in_flight
        if mask.any():
            events[label] = float(t[mask].min())

    return events


# ---------------------------------------------------------------------------
# Public summary computation
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> Dict:
    """Compute key flight metrics from a standardised DataFrame."""
    t   = _num(df, "t_flight_s")
    alt = _num(df, "alt_agl_m")
    vel = _num(df, "v_up_mps")
    acc = _num(df, "a_up_mps2")

    in_flight = t >= 0

    def _where(series: pd.Series) -> pd.Series:
        return series.where(in_flight)

    # Apogee
    valid_alt = _where(alt)
    apogee_idx = valid_alt.idxmax() if valid_alt.notna().any() else None
    apogee_m   = float(valid_alt[apogee_idx]) if apogee_idx is not None else float("nan")
    apogee_t   = float(t[apogee_idx])         if apogee_idx is not None else float("nan")

    # Max velocity — use direct measurement if reliable, else integrate accel.
    # TeleMega v_up_mps is GPS-derived and mostly absent during boost;
    # integrating a_up_mps2 from t=0 gives a better estimate in that case.
    valid_vel = vel.where(in_flight)
    vel_reliable = (
        valid_vel.notna().mean() > 0.5
        and float(valid_vel.max()) > 50.0
    )
    if vel_reliable:
        max_vel = float(valid_vel.max())
    else:
        valid_acc = _where(acc)
        if valid_acc.notna().any():
            t_f  = t[in_flight].reset_index(drop=True)
            a_f  = valid_acc[in_flight].reset_index(drop=True)
            dt_f = t_f.diff().fillna(0).clip(upper=0.5)
            v_int = (a_f * dt_f).cumsum()
            max_vel = float(v_int.max()) if v_int.notna().any() else float("nan")
        else:
            max_vel = float("nan")

    # Max acceleration (G)
    valid_acc = _where(acc)
    max_acc_g = float(valid_acc.max()) / _G if valid_acc.notna().any() else float("nan")

    # Burn time: last row in "boost" state
    burn_t = float("nan")
    if "state" in df.columns:
        state_s = df["state"].astype(str).str.lower().str.strip()
        in_boost = (state_s == "boost") & in_flight
        if in_boost.any():
            burn_t = float(t[in_boost].max())

    return {
        "apogee_m":           apogee_m,
        "apogee_ft":          apogee_m * _M_TO_FT,
        "time_to_apogee_s":   apogee_t,
        "max_velocity_mps":   max_vel,
        "max_velocity_fps":   max_vel * _MPS_TO_FPS,
        "max_accel_g":        max_acc_g,
        "burn_time_s":        burn_t,
    }


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def plot_flight(
    datasets: Dict[str, pd.DataFrame],
    flight_name: str,
    figures_dir: Path,
) -> List[Path]:
    """
    Generate a 3-panel flight-profile figure (altitude / velocity / acceleration)
    and save it to figures_dir/flight_profile.png.
    Returns list of saved paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Acceleration panel only when TeleMega data is present (has a_up_mps2)
    has_accel = (
        "telemega_std" in datasets
        and _num(datasets["telemega_std"], "a_up_mps2").notna().any()
    )
    n_panels = 3 if has_accel else 2
    heights = [2.5] * n_panels
    heights[0] = 3.0  # a bit more room for altitude

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(11, sum(heights) + 0.5 * (n_panels - 1)),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    ax_alt: plt.Axes = axes[0]
    ax_vel: plt.Axes = axes[1]
    ax_acc: plt.Axes | None = axes[2] if n_panels == 3 else None

    # ---- Collect events from primary source (TeleMega preferred) ----
    events: Dict[str, float] = {}
    for key in ("telemega_std", "blueraven_std", "featherweight_std"):
        if key in datasets:
            events = _detect_events(datasets[key])
            if events:
                break

    # ---- Plot altitude and velocity (skip IMU-only sources) ----
    for key, df in sorted(datasets.items()):
        if key not in _PLOT_SOURCES:
            continue
        t   = _num(df, "t_flight_s")
        alt = _num(df, "alt_agl_m")
        vel = _velocity_for_plot(df)
        # Only draw if source has meaningful altitude or velocity data
        if alt.dropna().empty and vel.dropna().empty:
            continue
        color = _SOURCE_COLOR.get(key, "gray")
        label = _SOURCE_LABEL.get(key, key)
        mask  = t.notna() & (t >= -10)

        ax_alt.plot(t[mask], alt[mask], color=color, lw=1.5, label=label, zorder=3)
        ax_vel.plot(t[mask], vel[mask], color=color, lw=1.5, label=label, zorder=3)

    if ax_acc is not None and "telemega_std" in datasets:
        df_tm = datasets["telemega_std"]
        t   = _num(df_tm, "t_flight_s")
        acc = _num(df_tm, "a_up_mps2")
        mask = t.notna() & (t >= -10)
        ax_acc.plot(
            t[mask], acc[mask] / _G,
            color=_SOURCE_COLOR["telemega_std"], lw=1.5, label="TeleMega", zorder=3,
        )
        # BlueRaven HR: also carries a_up_mps2 (from Accel_Y * g)
        if "blueraven_hr_std" in datasets:
            df_hr = datasets["blueraven_hr_std"]
            t_hr  = _num(df_hr, "t_flight_s")
            acc_hr = _num(df_hr, "a_up_mps2")
            mask_hr = t_hr.notna() & (t_hr >= -10) & acc_hr.notna()
            if mask_hr.any():
                ax_acc.plot(
                    t_hr[mask_hr], acc_hr[mask_hr] / _G,
                    color=_SOURCE_COLOR["blueraven_hr_std"], lw=0.6, alpha=0.6,
                    label="Blue Raven HR", zorder=2,
                )
        ax_acc.axhline(0, color="k", lw=0.6, ls="--", zorder=2)

    # ---- Event markers ----
    for ev_name, ev_t in sorted(events.items(), key=lambda x: x[1]):
        ec = _EVENT_COLOR.get(ev_name, "#333333")
        for ax in axes:
            ax.axvline(ev_t, color=ec, lw=1.0, ls="--", alpha=0.75, zorder=4)
        # Label at top of altitude panel using axes-coordinate y so it never clips
        ax_alt.text(
            ev_t, 1.0, f" {ev_name}",
            transform=ax_alt.get_xaxis_transform(),
            color=ec, fontsize=7.5, va="top", ha="left", rotation=90,
            clip_on=False,
        )

    # ---- Axis labels / styling ----
    ax_alt.set_title(flight_name, fontsize=11, pad=14)
    ax_alt.set_ylabel("Altitude AGL (m)")
    ax_alt.grid(True, alpha=0.3)
    ax_alt.legend(fontsize=9, loc="upper right")

    # Secondary y-axis in feet
    ax_alt_ft = ax_alt.twinx()
    ax_alt_ft.set_ylabel("Altitude AGL (ft)", fontsize=8.5, color="#444444")
    ax_alt_ft.tick_params(labelsize=8)

    ax_vel.set_ylabel("Vertical velocity (m/s)")
    ax_vel.axhline(0, color="k", lw=0.6, ls="--", zorder=2)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(fontsize=9, loc="upper right")

    # Secondary y-axis in fps
    ax_vel_fps = ax_vel.twinx()
    ax_vel_fps.set_ylabel("Vertical velocity (ft/s)", fontsize=8.5, color="#444444")
    ax_vel_fps.tick_params(labelsize=8)

    if ax_acc is not None:
        ax_acc.set_ylabel("Acceleration (G)")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(fontsize=9, loc="upper right")
        ax_acc.set_xlabel("Flight time (s)")
    else:
        ax_vel.set_xlabel("Flight time (s)")

    # ---- Sync twin-axis limits (must be done after data is plotted) ----
    ax_alt_ft.set_ylim([v * _M_TO_FT for v in ax_alt.get_ylim()])
    ax_vel_fps.set_ylim([v * _MPS_TO_FPS for v in ax_vel.get_ylim()])

    fig.savefig(figures_dir / "flight_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [figures_dir / "flight_profile.png"]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyze_flight(flight_dir: Path, verbose: bool = False) -> Dict:
    """
    Load interim CSVs for a flight, compute summary metrics, and save plots.
    Returns the summary dict.
    """
    interim = flight_dir / "interim"
    figures_dir = flight_dir / "figures"
    ensure_dir(figures_dir)

    datasets = _load_interim(interim)
    if not datasets:
        raise FileNotFoundError(f"No *_std.csv files found in {interim}")

    if verbose:
        for k in datasets:
            print(f"  Loaded {k}: {len(datasets[k])} rows")

    # Primary source for summary metrics: TeleMega > BlueRaven > FeatherWeight
    primary = None
    for key in ("telemega_std", "blueraven_std", "featherweight_std"):
        if key in datasets:
            primary = datasets[key]
            break
    if primary is None:
        primary = next(iter(datasets.values()))
    summary = compute_summary(primary)
    summary["flight"]  = flight_dir.name
    summary["sources"] = list(datasets.keys())

    saved = plot_flight(datasets, flight_dir.name, figures_dir)
    summary["figures"] = [str(p) for p in saved]

    return summary
