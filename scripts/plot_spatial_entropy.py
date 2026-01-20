#!/usr/bin/env python
"""
Publication-quality information-theoretic analysis of DLA aggregates.
CORRECTED VERSION - Fixed log base consistency for D₁ calculation.

Key Fix: D₁ now uses log₂ consistently with Shannon entropy.
Previously mixed log₂ (entropy) with ln (fitting), inflating D₁ by ~1.44.

Author: Sandy H. S. Herho
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from netCDF4 import Dataset
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
FIGS_DIR = Path(__file__).parent.parent / "figs"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

CASES = [
    ("case_1_classic_dla.nc", "Classic DLA"),
    ("case_2_multiple_seeds.nc", "Multiple Seeds"),
    ("case_3_radial_injection.nc", "Radial Injection"),
    ("case_4_high_density.nc", "High Density"),
]

FIG_WIDTH, FIG_HEIGHT, DPI = 12, 10, 300
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
LINEWIDTHS = [2.5, 2.5, 2.5, 2.5]
MARKERS = ['o', 's', '^', 'D']
MARKERSIZE = 6


def load_netcdf(filepath):
    with Dataset(filepath, 'r') as nc:
        return {
            'grid': nc.variables['grid'][:],
            'n_particles': nc.n_particles,
            'n_aggregates': nc.n_aggregates,
            'fractal_dimension': nc.fractal_dimension,
            'lattice_size': nc.lattice_size,
            'n_seeds': nc.n_seeds,
            'scenario': nc.scenario,
        }


def compute_box_statistics(grid, box_size):
    N = grid.shape[0]
    agg_mask = (grid == 2).astype(np.float64)
    n_boxes = N // box_size
    if n_boxes == 0:
        return 0, np.array([]), np.array([])
    
    truncated = agg_mask[:n_boxes * box_size, :n_boxes * box_size]
    masses = truncated.reshape(n_boxes, box_size, n_boxes, box_size).sum(axis=(1, 3)).flatten()
    n_occupied = np.sum(masses > 0)
    total_mass = masses.sum()
    probs = masses[masses > 0] / total_mass if total_mass > 0 else np.array([])
    return n_occupied, masses, probs


def compute_shannon_entropy(p):
    if len(p) == 0: return 0.0
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def compute_lacunarity(masses):
    if len(masses) == 0 or masses.sum() == 0: return 0.0
    mean_m = masses.mean()
    return (masses**2).mean() / (mean_m**2) - 1 if mean_m > 0 else 0.0


def compute_renyi_entropy(p, q):
    if len(p) == 0: return 0.0
    p = p[p > 0]
    if abs(q - 1) < 1e-10: return compute_shannon_entropy(p)
    if q == 0: return np.log2(len(p))
    return np.log2(np.sum(p**q)) / (1 - q)


def multiscale_analysis(grid):
    N = grid.shape[0]
    box_sizes = np.unique(np.logspace(np.log10(2), np.log10(N // 4), 20).astype(int))
    
    results = {k: [] for k in ['box_sizes', 'entropies', 'entropies_q2', 'lacunarities', 'n_occupied']}
    
    for eps in box_sizes:
        if eps < 2 or eps > N // 2: continue
        n_occ, masses, probs = compute_box_statistics(grid, eps)
        if len(probs) > 0:
            results['box_sizes'].append(eps)
            results['entropies'].append(compute_shannon_entropy(probs))
            results['entropies_q2'].append(compute_renyi_entropy(probs, 2))
            results['lacunarities'].append(compute_lacunarity(masses))
            results['n_occupied'].append(n_occ)
    
    return {k: np.array(v) for k, v in results.items()}


def fit_box_counting_dimension(box_sizes, n_occupied):
    valid = (box_sizes > 0) & (n_occupied > 0)
    if np.sum(valid) < 3: return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.log(box_sizes[valid]), np.log(n_occupied[valid]))
    return {'D0': -slope, 'D0_err': std_err, 'R2': r_value**2, 'p_value': p_value, 'intercept': intercept}


def fit_information_dimension(box_sizes, entropies):
    """FIXED: Use log₂ consistently with Shannon entropy."""
    valid = (box_sizes > 0) & (entropies > 0)
    if np.sum(valid) < 3: return None
    # CRITICAL FIX: log₂ not ln
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.log2(1 / box_sizes[valid]), entropies[valid])
    return {'D1': slope, 'D1_err': std_err, 'R2': r_value**2, 'p_value': p_value, 'intercept': intercept}


def fit_correlation_dimension(box_sizes, entropies_q2):
    valid = (box_sizes > 0) & (entropies_q2 > 0)
    if np.sum(valid) < 3: return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.log2(1 / box_sizes[valid]), entropies_q2[valid])
    return {'D2': slope, 'D2_err': std_err, 'R2': r_value**2, 'p_value': p_value, 'intercept': intercept}


def compute_statistics(all_data, all_results, all_fits):
    lines = ["=" * 75, "INFORMATION-THEORETIC ANALYSIS REPORT (CORRECTED)", "=" * 75, "",
             "KEY FIX: D₁ computed with log₂ consistently (was mixing log₂ and ln).", ""]
    
    lines.extend(["=" * 75, "DIMENSION ESTIMATES SUMMARY", "=" * 75, "",
                  f"  {'Case':<25} {'D_f':>8} {'D₀':>8} {'D₁':>8} {'D₂':>8}", "  " + "-" * 60])
    
    for i, (fn, name) in enumerate(CASES):
        d, f = all_data[i], all_fits[i]
        D0 = f['D0']['D0'] if f['D0'] else np.nan
        D1 = f['D1']['D1'] if f['D1'] else np.nan
        D2 = f['D2']['D2'] if f['D2'] else np.nan
        lines.append(f"  {name:<25} {d['fractal_dimension']:>8.4f} {D0:>8.4f} {D1:>8.4f} {D2:>8.4f}")
    
    lines.extend(["", "  For monofractals: D₀ ≈ D₁ ≈ D₂ ≈ D_f ≈ 1.71", ""])
    
    for i, (fn, name) in enumerate(CASES):
        d, r, f = all_data[i], all_results[i], all_fits[i]
        lines.extend(["=" * 75, f"CASE {i+1}: {name.upper()}", "=" * 75, "",
                      f"  Particles: {d['n_particles']:,}  |  D_f: {d['fractal_dimension']:.4f}", ""])
        if f['D0']:
            lines.append(f"  D₀ = {f['D0']['D0']:.4f} ± {f['D0']['D0_err']:.4f} (R² = {f['D0']['R2']:.4f})")
        if f['D1']:
            lines.append(f"  D₁ = {f['D1']['D1']:.4f} ± {f['D1']['D1_err']:.4f} (R² = {f['D1']['R2']:.4f}) [CORRECTED]")
        if f['D2']:
            lines.append(f"  D₂ = {f['D2']['D2']:.4f} ± {f['D2']['D2_err']:.4f} (R² = {f['D2']['R2']:.4f})")
        lines.append("")
    
    lines.extend(["=" * 75, "END OF REPORT", "=" * 75])
    return "\n".join(lines)


def setup_figure_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11, 'font.weight': 'bold',
        'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.linewidth': 1.5,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'axes.grid': True, 'grid.alpha': 0.3,
    })


def create_figure(all_data, all_results, all_fits):
    setup_figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    axes = axes.flatten()
    legend_lines, legend_labels = [], []
    
    # (a) Shannon Entropy
    ax = axes[0]
    for i, (_, name) in enumerate(CASES):
        r = all_results[i]
        line, = ax.plot(r['box_sizes'], r['entropies'], color=COLORS[i], lw=LINEWIDTHS[i],
                       marker=MARKERS[i], ms=MARKERSIZE, mfc='white', mew=1.5)
        legend_lines.append(line); legend_labels.append(name)
    ax.set_xlabel('Box size ε', fontweight='bold'); ax.set_ylabel('H(ε) [bits]', fontweight='bold')
    ax.set_xscale('log'); ax.text(0.5, 1.05, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', ha='center')
    
    # (b) Lacunarity
    ax = axes[1]
    for i, _ in enumerate(CASES):
        r = all_results[i]
        ax.plot(r['box_sizes'], r['lacunarities'], color=COLORS[i], lw=LINEWIDTHS[i],
               marker=MARKERS[i], ms=MARKERSIZE, mfc='white', mew=1.5)
    ax.set_xlabel('Box size ε', fontweight='bold'); ax.set_ylabel('Λ(ε)', fontweight='bold')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.text(0.5, 1.05, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold', ha='center')
    
    # (c) Box-counting D₀
    ax = axes[2]
    for i, _ in enumerate(CASES):
        r, f = all_results[i], all_fits[i]
        log_eps, log_N = np.log(r['box_sizes']), np.log(r['n_occupied'])
        ax.scatter(log_eps, log_N, c=COLORS[i], s=40, marker=MARKERS[i], alpha=0.7, edgecolors='white')
        if f['D0']:
            x = np.linspace(log_eps.min(), log_eps.max(), 100)
            ax.plot(x, -f['D0']['D0'] * x + f['D0']['intercept'], color=COLORS[i], lw=2, ls='--', alpha=0.8)
    ax.set_xlabel('log ε', fontweight='bold'); ax.set_ylabel('log N(ε)', fontweight='bold')
    ax.text(0.5, 1.05, '(c)', transform=ax.transAxes, fontsize=16, fontweight='bold', ha='center')
    for i, (_, name) in enumerate(CASES):
        if all_fits[i]['D0']:
            ax.text(0.98, 0.95 - 0.08*i, f"D₀ = {all_fits[i]['D0']['D0']:.2f}",
                   transform=ax.transAxes, fontsize=9, fontweight='bold', color=COLORS[i], ha='right', va='top')
    
    # (d) Information D₁ (CORRECTED)
    ax = axes[3]
    for i, _ in enumerate(CASES):
        r, f = all_results[i], all_fits[i]
        log2_inv = np.log2(1 / r['box_sizes'])  # FIXED: log₂
        ax.scatter(log2_inv, r['entropies'], c=COLORS[i], s=40, marker=MARKERS[i], alpha=0.7, edgecolors='white')
        if f['D1']:
            x = np.linspace(log2_inv.min(), log2_inv.max(), 100)
            ax.plot(x, f['D1']['D1'] * x + f['D1']['intercept'], color=COLORS[i], lw=2, ls='--', alpha=0.8)
    ax.set_xlabel('log₂(1/ε)', fontweight='bold'); ax.set_ylabel('H(ε) [bits]', fontweight='bold')
    ax.text(0.5, 1.05, '(d)', transform=ax.transAxes, fontsize=16, fontweight='bold', ha='center')
    for i, (_, name) in enumerate(CASES):
        if all_fits[i]['D1']:
            ax.text(0.02, 0.95 - 0.08*i, f"D₁ = {all_fits[i]['D1']['D1']:.2f}",
                   transform=ax.transAxes, fontsize=9, fontweight='bold', color=COLORS[i], ha='left', va='top')
    
    for ax in axes:
        for spine in ax.spines.values(): spine.set_linewidth(1.5)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98], h_pad=2.5, w_pad=2.0)
    leg = fig.legend(legend_lines, legend_labels, loc='lower center', ncol=4, fontsize=11,
                    frameon=True, fancybox=False, edgecolor='black', bbox_to_anchor=(0.5, 0.01),
                    handlelength=3, handletextpad=0.8, columnspacing=2.0)
    for t in leg.get_texts(): t.set_fontweight('bold')
    return fig


def main():
    print("=" * 65)
    print("Spatial Entropy Analysis (CORRECTED)")
    print("=" * 65)
    print("\nFix: D₁ now uses log₂ consistently with Shannon entropy\n")
    
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_data = [load_netcdf(DATA_DIR / fn) for fn, _ in CASES]
    all_results = [multiscale_analysis(d['grid']) for d in all_data]
    all_fits = [{'D0': fit_box_counting_dimension(r['box_sizes'], r['n_occupied']),
                 'D1': fit_information_dimension(r['box_sizes'], r['entropies']),
                 'D2': fit_correlation_dimension(r['box_sizes'], r['entropies_q2'])}
                for r in all_results]
    
    print("Corrected dimensions:")
    for i, (_, name) in enumerate(CASES):
        f = all_fits[i]
        print(f"  {name}: D₀={f['D0']['D0']:.3f}, D₁={f['D1']['D1']:.3f}, D₂={f['D2']['D2']:.3f}")
    
    fig = create_figure(all_data, all_results, all_fits)
    fig.savefig(FIGS_DIR / "spatial_entropy.png", dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGS_DIR / "spatial_entropy.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    with open(REPORTS_DIR / "spatial_entropy_report.txt", 'w') as f:
        f.write(compute_statistics(all_data, all_results, all_fits))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
