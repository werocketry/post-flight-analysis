from __future__ import annotations
from pathlib import Path

import typer

from .common import PreprocessOptions, ensure_dir, write_meta
from .parsers import (
    parse_telemega_csv,
    parse_blueraven_csv,
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
    """Return 'telemega' | 'blueraven' | None by inspecting the header line."""
    if not p.is_file() or p.suffix.lower() != ".csv":
        return None
    head = _read_first_nonempty_line(p)
    hlow = head.lower()
    if hlow.startswith("#version,serial,flight"):
        return "telemega"
    if "baro_altitude_asl_(feet)" in hlow or "baro_altitude_agl_(feet)" in hlow:
        return "blueraven"
    return None


def _find_files(raw: Path, verbose: bool) -> dict[str, Path]:
    kinds: dict[str, Path] = {}
    for p in sorted(raw.glob("*")):
        kind = _sniff_kind(p)
        if kind:
            if verbose:
                typer.echo(f"Detected {kind} file: {p.name}")
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

    if not tele_file and not br_file:
        typer.echo("No TeleMega or Blue Raven CSVs detected after sniffing.")
        raise typer.Exit(code=3)
