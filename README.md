# post-flight-analysis

This repository hosts flight data and analysis for our rocketry team.

It is designed to support raw log storage, parsing, normalization, and analysis across multiple years and altimeter devices.

The long-term goal is to evolve from simple scripts into a reusable CLI and eventually a library for other teams.

## Structure

- **data/** — raw and processed flight logs, organized by year and flight
- **scripts/** — simple scripts to parse and normalize device logs
- **analysis/** — exploratory scripts and reports
- **config/** — device- and pipeline-specific settings
- **notebooks/** — Jupyter notebooks for ad-hoc exploration
- **docs/** — project documentation

## Data Organization

Each flight lives in a directory:

```sh
data/{year}/{FLIGHT-ID}/
├─ metadata/ # flight.yaml with contextual information
├─ raw/ # original altimeter logs, untouched
├─ interim/ # parser outputs, intermediate format
└─ processed/ # normalized tables, figures, reports
```

## Devices

Currently supported:

- **Blue Raven**
- **TeleMega**

Future devices can be added through `config/devices/*.yaml` and dedicated parsers.

## License

- **Code**: Apache-2.0
- **Data**: CC BY 4.0  
  See `LICENSES/` for details.
