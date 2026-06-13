from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from dateutil import tz, parser as dtparser

SCHEMA_VERSION = "gfr-0.1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_meta(
    out_json: Path,
    *,
    input_files: Iterable[Path],
    device: str,
    serial: Optional[str],
    firmware: Optional[str],
    mapping_used: Dict[str, str],
    units_converted: Iterable[str],
    liftoff_method: str,
    row_counts: Dict[str, int],
    extra: Optional[Dict] = None,
) -> None:
    payload = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "pfa_version": "0.1.0",
        "schema_version": SCHEMA_VERSION,
        "device": {"type": device, "serial": serial, "firmware": firmware},
        "input_files": [
            {"path": str(p), "sha256": sha256_file(p)} for p in input_files
        ],
        "mapping_used": mapping_used,
        "units_converted": list(units_converted),
        "liftoff_method": liftoff_method,
        "row_counts": row_counts,
    }
    if extra:
        payload.update(extra)
    out_json.write_text(json.dumps(payload, indent=2))


def local_to_utc(dt_str: str, tz_name: str) -> float:
    """
    Parse a local naive date-time string and return epoch seconds in UTC.
    dt_str: ISO-like string (e.g., '2025-08-21 13:28:45.051').
    tz_name: IANA TZ, e.g., 'America/Toronto'.
    """
    local = dtparser.parse(dt_str)
    if local.tzinfo is None:
        local = local.replace(tzinfo=tz.gettz(tz_name))
    return local.astimezone(tz.UTC).timestamp()


def feet_to_m(x: pd.Series | float) -> pd.Series | float:
    return x * 0.3048


def g_to_mps2(x: pd.Series | float) -> pd.Series | float:
    return x * 9.80665


def coalesce_cols(df: pd.DataFrame, dst: str, candidates: Iterable[str]) -> None:
    for c in candidates:
        if c in df.columns:
            df[dst] = df[c]
            return
    df[dst] = pd.NA


@dataclass
class PreprocessOptions:
    tz_name: str
