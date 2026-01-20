# dla-paper-scripts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Requirements

```bash
pip install numpy matplotlib netCDF4 scipy
```

## Usage

```bash
cd scripts
python plot_spatiotemporal.py    # Fig 1: 4×3 growth snapshots
python plot_mass_radius.py       # Fig 2: Fractal dimension M(R)~R^D
python plot_growth_dynamics.py   # Fig 3: Growth curves N(t)
python plot_spatial_entropy.py   # Fig 4: Shannon entropy & lacunarity
```

Outputs: `../figs/` (PNG/PDF/EPS) and `../reports/` (TXT).

## Test Cases

| Case | Description |
|------|-------------|
| 1 | Classic DLA, single central seed |
| 2 | Multiple seeds (12), competitive growth |
| 3 | Radial injection geometry |
| 4 | High walker density |

## Related

Main library: [dla-ideal-solver](https://github.com/sandyherho/dla-ideal-solver)


