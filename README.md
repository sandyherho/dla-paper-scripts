# `dla-paper-scripts`: Supplementary Analysis Scripts

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Supplementary visualization and statistical analysis scripts for the paper:

> Herho, S. H. S., Fajary, F. R., Trilaksono, N. J., Anwar, I. P., Suwarman, R., & Irawan, D. E. (2026). Diffusion-Limited Aggregation on Two-Dimensional Lattices: A Numba-Accelerated Implementation. *Mathematical Modeling and Computing*, *[Volume]*, [Pages]. https://doi.org/[DOI]

## Data & Outputs

All simulation data (NetCDF), figures (PNG/PDF/EPS), and statistical reports (TXT) are available at:

📦 **OSF Repository:** [https://doi.org/10.17605/OSF.IO/XXXXX](https://doi.org/10.17605/OSF.IO/XXXXX)

## Main Library

These scripts accompany the `dla-ideal-solver` Python library:

- **GitHub:** [https://github.com/sandyherho/dla-ideal-solver](https://github.com/sandyherho/dla-ideal-solver)
- **PyPI:** [https://pypi.org/project/dla-ideal-solver/](https://pypi.org/project/dla-ideal-solver/)

Install via:
```bash
pip install dla-ideal-solver
```

## Repository Structure

```
dla-paper-scripts/
├── data/           # NetCDF simulation outputs (4 cases)
├── figs/           # Generated figures (PNG, PDF, EPS)
├── reports/        # Statistical analysis reports (TXT)
└── scripts/
    ├── plot_spatiotemporal.py       # Fig 1: Growth snapshots (4×3 panel)
    ├── plot_mass_radius.py          # Fig 2: Fractal scaling M(R) ~ R^D
    ├── plot_growth_dynamics.py      # Fig 3: Growth curves and rates
    └── plot_spatial_entropy.py      # Fig 4: Information-theoretic analysis
```

## Scripts

| Script | Description | Output Figure |
|--------|-------------|---------------|
| `plot_spatiotemporal.py` | 4×3 panel showing aggregate growth snapshots at beginning, middle, and end times | `spatiotemporal_snapshots.*` |
| `plot_mass_radius.py` | Log-log mass-radius plots for fractal dimension extraction with linear regression | `mass_radius_scaling.*` |
| `plot_growth_dynamics.py` | Particle accumulation curves and instantaneous growth rates | `growth_dynamics.*` |
| `plot_spatial_entropy.py` | Shannon entropy and lacunarity analysis of aggregate structure | `spatial_entropy.*` |

## Usage

```bash
cd scripts

# Generate individual figures
python plot_spatiotemporal.py
python plot_mass_radius.py
python plot_growth_dynamics.py
python plot_spatial_entropy.py
```

Each script:
- Reads NetCDF data from `../data/`
- Saves figures to `../figs/` (PNG 300 dpi, PDF, EPS)
- Saves statistical reports to `../reports/` (TXT)

## Test Cases

| Case | File | Description | Seeds | Walkers |
|------|------|-------------|-------|---------|
| 1 | `case_1_classic_dla.nc` | Classic DLA with single central seed | 1 | 10,000 |
| 2 | `case_2_multiple_seeds.nc` | Competitive growth from 12 random seeds | 12 | 15,000 |
| 3 | `case_3_radial_injection.nc` | Controlled radial injection geometry | 1 | 10,000 |
| 4 | `case_4_high_density.nc` | High walker density regime | 1 | 25,000 |

## Dependencies

```
numpy>=1.20.0
matplotlib>=3.3.0
netCDF4>=1.5.0
scipy>=1.7.0
```

Install all dependencies:
```bash
pip install numpy matplotlib netCDF4 scipy
```

## Mathematical Background

### Fractal Dimension

The fractal dimension $D_f$ is extracted from the mass-radius scaling:

$$M(R) \sim R^{D_f}$$

Linear regression on log-transformed data yields:

$$\log M(R) = D_f \log R + \text{const}$$

For 2D DLA, theory predicts $D_f \approx 1.71$.

### Spatial Entropy

The Shannon entropy of the spatial distribution quantifies aggregate compactness:

$$H = -\sum_{i} p_i \log_2 p_i$$

where $p_i$ is the occupation probability in cell $i$.

### Lacunarity

Lacunarity $\Lambda$ measures the "gappiness" or heterogeneity of the fractal:

$$\Lambda(r) = \frac{\langle M^2 \rangle - \langle M \rangle^2}{\langle M \rangle^2}$$

computed over boxes of size $r$.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Authors

- Sandy H. S. Herho (sandy.herho@ronininstitute.org)
- Faiz R. Fajary
- Nurjanna J. Trilaksono
- Iwan P. Anwar
- Rusmawan Suwarman
- Dasapta E. Irawan

## Citation

```bibtex
@software{dla_paper_scripts_2026,
  author = {Herho, Sandy H. S. and Fajary, Faiz R. and 
            Trilaksono, Nurjanna J. and Anwar, Iwan P. and 
            Suwarman, Rusmawan and Irawan, Dasapta E.},
  title = {DLA Paper Scripts: Supplementary Analysis for Diffusion-Limited Aggregation},
  year = {2026},
  url = {https://github.com/sandyherho/dla-paper-scripts}
}
```
