from __future__ import annotations
from pathlib import Path

import typer

from .common import PreprocessOptions, ensure_dir, write_meta
from .parsers import (
    parse_telemega_csv,
    parse_blueraven_csv,
    parse_blueraven_hr_csv,
    parse_featherweight,
)

app = typer.Typer(add_completion=False)


@app.callback()
def _callback() -> None:
    """Post-flight analysis tools."""


def _read_first_nonempty_line(p: Path) -> str:
    with p.open("rb") as f:
        raw = f.read(4096)
    try:
        text = raw.decode("utf-8-sig", errors="ignore")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _sniff_kind(p: Path) -> str | None:
    """Return kind string by inspecting the file header/content."""
    if not p.is_file():
        return None
    suffix = p.suffix.lower()

    if suffix == ".csv":
        head = _read_first_nonempty_line(p)
        hlow = head.lower()
        if hlow.startswith("#version,serial,flight"):
            return "telemega"
        if "baro_altitude_asl_(feet)" in hlow or "baro_altitude_agl_(feet)" in hlow:
            return "blueraven"
        if "gyro_x" in hlow and "flight_time_(s)" in hlow:
            return "blueraven_hr"
        if "tracker lat" in hlow and "tracker lon" in hlow:
            return "featherweight"
        return None

    if suffix == ".xlsx":
        try:
            import openpyxl
            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            ws = wb.active
            first_row = next(ws.iter_rows(max_row=1, values_only=True), ())
            hlow = ",".join(
                str(c).lower().strip() for c in first_row if c is not None
            )
            if "tracker lat" in hlow and "tracker lon" in hlow:
                return "featherweight"
        except Exception:
            pass
        return None

    return None


def _find_files(raw: Path, verbose: bool) -> dict[str, Path]:
    kinds: dict[str, Path] = {}
    # Top-level raw/ has highest priority
    for p in sorted(raw.glob("*")):
        if p.is_dir():
            continue
        kind = _sniff_kind(p)
        if kind:
            if verbose:
                typer.echo(f"Detected {kind} file: {p.name}")
            kinds[kind] = p
    # raw/src/ fills in any kinds not already found
    src_dir = raw / "src"
    if src_dir.is_dir():
        for p in sorted(src_dir.glob("*")):
            if p.is_dir():
                continue
            kind = _sniff_kind(p)
            if kind and kind not in kinds:
                if verbose:
                    typer.echo(f"Detected {kind} file (src/): {p.name}")
                kinds[kind] = p
    return kinds


@app.command()
def preprocess(
    flight_dir: str = typer.Argument(..., help="Path to data/<date>_<vehicle>_<event>"),
    tz: str = typer.Option(
        "America/Toronto", "--tz", help="Local time zone for Blue Raven logs"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    fpath = Path(flight_dir)
    raw = fpath / "raw"
    interim = fpath / "interim"
    ensure_dir(interim)

    if not raw.exists():
        typer.echo(f"Raw folder not found: {raw}")
        raise typer.Exit(code=1)

    kinds = _find_files(raw, verbose)
    if not kinds:
        typer.echo(f"No recognizable device files found in {raw}")
        raise typer.Exit(code=2)

    opts = PreprocessOptions(tz_name=tz)

    tele_file = kinds.get("telemega")
    if tele_file:
        if verbose:
            typer.echo("Parsing TeleMega...")
        df, meta = parse_telemega_csv(tele_file)
        out_csv = interim / "telemega_std.csv"
        df.to_csv(out_csv, index=False)
        gps_valid = int(df["lat_deg"].notna().sum()) if "lat_deg" in df.columns else 0
        write_meta(
            interim / "telemega_std.meta.json",
            input_files=[tele_file],
            device="telemega",
            serial=meta.get("serial"),
            firmware=meta.get("firmware"),
            mapping_used=meta.get("mapping_used", {}),
            units_converted=meta.get("units_converted", []),
            liftoff_method=meta.get("liftoff_method", "unknown"),
            row_counts=meta.get(
                "row_counts",
                {
                    "total": len(df),
                    "after_clean": len(df),
                    "gps_valid": gps_valid,
                },
            ),
        )
        typer.echo(f"Wrote {out_csv}")

    br_file = kinds.get("blueraven")
    if br_file:
        if verbose:
            typer.echo("Parsing Blue Raven altimeter...")
        df, meta = parse_blueraven_csv(br_file, tz_name=opts.tz_name)
        out_csv = interim / "blueraven_std.csv"
        df.to_csv(out_csv, index=False)
        gps_valid = int(df["lat_deg"].notna().sum()) if "lat_deg" in df.columns else 0
        write_meta(
            interim / "blueraven_std.meta.json",
            input_files=[br_file],
            device="blueraven",
            serial=meta.get("serial"),
            firmware=meta.get("firmware"),
            mapping_used=meta.get("mapping_used", {}),
            units_converted=meta.get("units_converted", []),
            liftoff_method=meta.get("liftoff_method", "unknown"),
            row_counts=meta.get(
                "row_counts",
                {
                    "total": len(df),
                    "after_clean": len(df),
                    "gps_valid": gps_valid,
                },
            ),
        )
        typer.echo(f"Wrote {out_csv}")

    fw_file = kinds.get("featherweight")
    if fw_file:
        if verbose:
            typer.echo("Parsing FeatherWeight GPS tracker...")
        df, meta = parse_featherweight(fw_file)
        out_csv = interim / "featherweight_std.csv"
        df.to_csv(out_csv, index=False)
        gps_valid = int(df["lat_deg"].notna().sum()) if "lat_deg" in df.columns else 0
        write_meta(
            interim / "featherweight_std.meta.json",
            input_files=[fw_file],
            device="featherweight",
            serial=meta.get("serial"),
            firmware=meta.get("firmware"),
            mapping_used=meta.get("mapping_used", {}),
            units_converted=meta.get("units_converted", []),
            liftoff_method=meta.get("liftoff_method", "unknown"),
            row_counts=meta.get(
                "row_counts",
                {"total": len(df), "after_clean": len(df), "gps_valid": gps_valid},
            ),
        )
        typer.echo(f"Wrote {out_csv}")

    br_hr_file = kinds.get("blueraven_hr")
    if br_hr_file:
        if verbose:
            typer.echo("Parsing Blue Raven HR IMU...")
        df, meta = parse_blueraven_hr_csv(br_hr_file, tz_name=opts.tz_name)
        out_csv = interim / "blueraven_hr_std.csv"
        df.to_csv(out_csv, index=False)
        write_meta(
            interim / "blueraven_hr_std.meta.json",
            input_files=[br_hr_file],
            device="blueraven_hr",
            serial=meta.get("serial"),
            firmware=meta.get("firmware"),
            mapping_used=meta.get("mapping_used", {}),
            units_converted=meta.get("units_converted", []),
            liftoff_method=meta.get("liftoff_method", "unknown"),
            row_counts=meta.get(
                "row_counts",
                {"total": len(df), "after_clean": len(df), "gps_valid": 0},
            ),
        )
        typer.echo(f"Wrote {out_csv}")

    if not any([tele_file, br_file, fw_file, br_hr_file]):
        typer.echo("No recognisable device files detected after sniffing.")
        raise typer.Exit(code=3)


@app.command()
def analyze(
    flight_dir: str = typer.Argument(..., help="Path to data/<date>_<vehicle>_<event>"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """
    Generate flight-profile plots and print summary metrics for a preprocessed flight.
    Reads interim/*_std.csv produced by 'pfa preprocess'.
    Saves figures to <flight_dir>/figures/.
    """
    import math
    from .analyze import analyze_flight

    fpath = Path(flight_dir)
    if not (fpath / "interim").exists():
        typer.echo(f"No interim/ directory found in {fpath}. Run 'pfa preprocess' first.")
        raise typer.Exit(code=1)

    summary = analyze_flight(fpath, verbose=verbose)

    def _fmt(val: float, fmt: str = ".0f") -> str:
        return (f"{val:{fmt}}") if not math.isnan(val) else "n/a"

    typer.echo(f"\n=== {summary['flight']} ===")
    typer.echo(f"  Sources:          {', '.join(summary['sources'])}")
    typer.echo(f"  Apogee AGL:       {_fmt(summary['apogee_m'])} m  ({_fmt(summary['apogee_ft'])} ft)")
    typer.echo(f"  Time to apogee:   {_fmt(summary['time_to_apogee_s'], '.1f')} s")
    typer.echo(f"  Max velocity:     {_fmt(summary['max_velocity_mps'])} m/s  ({_fmt(summary['max_velocity_fps'])} ft/s)")
    typer.echo(f"  Max acceleration: {_fmt(summary['max_accel_g'], '.1f')} G")
    if not math.isnan(summary["burn_time_s"]):
        typer.echo(f"  Motor burn time:  {_fmt(summary['burn_time_s'], '.2f')} s")
    for fig_path in summary["figures"]:
        typer.echo(f"  Saved figure:     {fig_path}")


@app.command()
def compare(
    data_dir: str = typer.Argument("data", help="Root directory containing flight folders"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """
    Print a cross-flight summary table for all preprocessed flights found under data_dir.
    Does not regenerate figures.
    """
    import math
    from .analyze import _load_interim, compute_summary

    data_path = Path(data_dir)
    flight_dirs = sorted(
        p for p in data_path.iterdir()
        if p.is_dir() and (p / "interim").exists()
    )

    if not flight_dirs:
        typer.echo(f"No preprocessed flights found under {data_path}")
        raise typer.Exit(code=1)

    summaries = []
    for fd in flight_dirs:
        try:
            datasets = _load_interim(fd / "interim")
            if not datasets:
                continue
            # Primary source: TeleMega > BlueRaven > FeatherWeight > HR
            primary = None
            for key in ("telemega_std", "blueraven_std", "featherweight_std"):
                if key in datasets:
                    primary = datasets[key]
                    break
            if primary is None:
                primary = next(iter(datasets.values()))
            s = compute_summary(primary)
            s["flight"] = fd.name
            s["sources"] = list(datasets.keys())
            summaries.append(s)
            if verbose:
                typer.echo(f"  Loaded {fd.name}: {list(datasets.keys())}")
        except Exception as exc:
            typer.echo(f"  Warning: {fd.name}: {exc}")

    if not summaries:
        typer.echo("No summaries could be computed.")
        raise typer.Exit(code=1)

    def _f(v: float, fmt: str = ".0f") -> str:
        return f"{v:{fmt}}" if not math.isnan(v) else "n/a"

    w = 34  # flight name column width
    typer.echo("")
    typer.echo(
        f"{'Flight':<{w}} {'Apogee(m)':>10} {'Apogee(ft)':>11} "
        f"{'t_apo(s)':>9} {'MaxVel(m/s)':>12} {'MaxG':>6} {'Burn(s)':>8}  Sources"
    )
    typer.echo("-" * (w + 10 + 11 + 9 + 12 + 6 + 8 + 30))
    for s in summaries:
        src_str = ", ".join(k.replace("_std", "") for k in s["sources"])
        typer.echo(
            f"{s['flight']:<{w}} "
            f"{_f(s['apogee_m']):>10} "
            f"{_f(s['apogee_ft']):>11} "
            f"{_f(s['time_to_apogee_s'], '.1f'):>9} "
            f"{_f(s['max_velocity_mps'], '.0f'):>12} "
            f"{_f(s['max_accel_g'], '.1f'):>6} "
            f"{_f(s['burn_time_s'], '.2f'):>8}  "
            f"{src_str}"
        )
    typer.echo("")

