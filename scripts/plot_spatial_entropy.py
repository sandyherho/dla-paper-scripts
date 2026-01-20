#!/usr/bin/env python
"""
Publication-quality information-theoretic analysis of DLA aggregates.

Creates a 2×2 panel figure:
  - (a) Shannon entropy H vs box size (all cases)
  - (b) Lacunarity Λ vs box size (all cases)
  - (c) Box-counting dimension D_b estimation
  - (d) Information dimension D_1 estimation

Information-Theoretic Measures:
  1. Shannon Entropy H(ε): Measures spatial disorder at scale ε
     H(ε) = -Σ p_i log₂(p_i) where p_i = N_i/N_total
     
  2. Lacunarity Λ(ε): Measures heterogeneity/gappiness
     Λ(ε) = ⟨M²⟩/⟨M⟩² - 1 over boxes of size ε
     
  3. Box-counting Dimension D_0: 
     N(ε) ~ ε^(-D_0)
     
  4. Information Dimension D_1:
     H(ε) ~ D_1 log(1/ε)

Output:
  - ../figs/spatial_entropy.png (300 dpi)
  - ../figs/spatial_entropy.pdf (vector)
  - ../figs/spatial_entropy.eps (vector)
  - ../reports/spatial_entropy_report.txt

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

# Figure settings
FIG_WIDTH = 12
FIG_HEIGHT = 10
DPI = 300

# Panel labels
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

# Colors for 4 cases
COLORS = [
    '#E63946',  # Red - Case 1
    '#457B9D',  # Blue - Case 2
    '#2A9D8F',  # Teal - Case 3
    '#F4A261',  # Orange - Case 4
]

LINEWIDTHS = [2.5, 2.5, 2.5, 2.5]
MARKERS = ['o', 's', '^', 'D']
MARKERSIZE = 6


# =============================================================================
# Data Loading
# =============================================================================

def load_netcdf(filepath):
    """Load NetCDF data and extract key variables."""
    with Dataset(filepath, 'r') as nc:
        data = {
            'grid': nc.variables['grid'][:],
            'n_particles': nc.n_particles,
            'n_aggregates': nc.n_aggregates,
            'fractal_dimension': nc.fractal_dimension,
            'center_x': nc.center_x,
            'center_y': nc.center_y,
            'lattice_size': nc.lattice_size,
            'n_walkers': nc.n_walkers,
            'n_seeds': nc.n_seeds,
            'scenario': nc.scenario,
        }
    return data


# =============================================================================
# Information-Theoretic Analysis Functions
# =============================================================================

def compute_box_statistics(grid, box_size):
    """
    Compute box statistics for a given box size.
    
    Returns:
        n_occupied: Number of boxes containing at least one particle
        masses: Array of particle counts in each box
        probabilities: Normalized occupation probabilities
    """
    N = grid.shape[0]
    agg_mask = (grid == 2).astype(np.float64)
    
    # Number of boxes in each dimension
    n_boxes = N // box_size
    if n_boxes == 0:
        return 0, np.array([]), np.array([])
    
    # Truncate grid to fit integer number of boxes
    truncated = agg_mask[:n_boxes * box_size, :n_boxes * box_size]
    
    # Reshape and sum to get box masses
    reshaped = truncated.reshape(n_boxes, box_size, n_boxes, box_size)
    masses = reshaped.sum(axis=(1, 3))
    
    # Flatten
    masses_flat = masses.flatten()
    
    # Number of occupied boxes
    n_occupied = np.sum(masses_flat > 0)
    
    # Probabilities (only for occupied boxes)
    total_mass = masses_flat.sum()
    if total_mass > 0:
        probabilities = masses_flat[masses_flat > 0] / total_mass
    else:
        probabilities = np.array([])
    
    return n_occupied, masses_flat, probabilities


def compute_shannon_entropy(probabilities):
    """
    Compute Shannon entropy: H = -Σ p_i log₂(p_i)
    
    Returns entropy in bits.
    """
    if len(probabilities) == 0:
        return 0.0
    
    # Filter out zeros
    p = probabilities[probabilities > 0]
    
    return -np.sum(p * np.log2(p))


def compute_lacunarity(masses):
    """
    Compute lacunarity: Λ = ⟨M²⟩/⟨M⟩² - 1
    
    Lacunarity measures the "gappiness" or heterogeneity of the distribution.
    Λ = 0 for uniform distribution
    Λ > 0 indicates heterogeneity (higher = more gaps)
    """
    if len(masses) == 0 or masses.sum() == 0:
        return 0.0
    
    mean_m = masses.mean()
    mean_m2 = (masses**2).mean()
    
    if mean_m == 0:
        return 0.0
    
    return mean_m2 / (mean_m**2) - 1


def multiscale_analysis(grid, box_sizes=None):
    """
    Perform multiscale information-theoretic analysis.
    
    Args:
        grid: 2D lattice array
        box_sizes: Array of box sizes to analyze
    
    Returns:
        dict with box_sizes, entropies, lacunarities, n_boxes
    """
    N = grid.shape[0]
    
    if box_sizes is None:
        # Generate logarithmically spaced box sizes
        min_size = 2
        max_size = N // 4
        n_sizes = 20
        box_sizes = np.unique(np.logspace(
            np.log10(min_size), np.log10(max_size), n_sizes
        ).astype(int))
    
    results = {
        'box_sizes': [],
        'entropies': [],
        'lacunarities': [],
        'n_occupied': [],
        'total_boxes': [],
    }
    
    for eps in box_sizes:
        if eps < 2 or eps > N // 2:
            continue
            
        n_occ, masses, probs = compute_box_statistics(grid, eps)
        
        if len(probs) > 0:
            H = compute_shannon_entropy(probs)
            L = compute_lacunarity(masses)
            
            results['box_sizes'].append(eps)
            results['entropies'].append(H)
            results['lacunarities'].append(L)
            results['n_occupied'].append(n_occ)
            results['total_boxes'].append(len(masses))
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results


def fit_box_counting_dimension(box_sizes, n_occupied):
    """
    Fit box-counting dimension: N(ε) ~ ε^(-D_0)
    
    log N = -D_0 log ε + const
    """
    valid = (box_sizes > 0) & (n_occupied > 0)
    if np.sum(valid) < 3:
        return None
    
    log_eps = np.log(box_sizes[valid])
    log_N = np.log(n_occupied[valid])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_N)
    
    return {
        'D0': -slope,
        'D0_err': std_err,
        'R2': r_value**2,
        'p_value': p_value,
        'intercept': intercept,
    }


def fit_information_dimension(box_sizes, entropies):
    """
    Fit information dimension: H(ε) ~ D_1 log(1/ε)
    
    H = D_1 log(1/ε) + const = -D_1 log(ε) + const
    
    So slope of H vs log(1/ε) gives D_1.
    """
    valid = (box_sizes > 0) & (entropies > 0)
    if np.sum(valid) < 3:
        return None
    
    log_inv_eps = np.log(1 / box_sizes[valid])
    H = entropies[valid]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_inv_eps, H)
    
    return {
        'D1': slope,
        'D1_err': std_err,
        'R2': r_value**2,
        'p_value': p_value,
        'intercept': intercept,
    }


def compute_renyi_entropy(probabilities, q):
    """
    Compute Rényi entropy of order q:
    H_q = (1/(1-q)) log₂(Σ p_i^q)
    
    Special cases:
    - q → 1: Shannon entropy
    - q = 0: log₂(support size)
    - q = 2: Collision entropy
    """
    if len(probabilities) == 0:
        return 0.0
    
    p = probabilities[probabilities > 0]
    
    if q == 1:
        return compute_shannon_entropy(p)
    elif q == 0:
        return np.log2(len(p))
    else:
        return np.log2(np.sum(p**q)) / (1 - q)


# =============================================================================
# Statistics Report
# =============================================================================

def compute_statistics(all_data, all_results, all_fits):
    """Compute comprehensive statistics and return formatted report."""
    lines = []
    lines.append("=" * 75)
    lines.append("INFORMATION-THEORETIC ANALYSIS REPORT")
    lines.append("=" * 75)
    lines.append("")
    lines.append("Multiscale information-theoretic analysis of DLA aggregates.")
    lines.append("")
    lines.append("MEASURES COMPUTED:")
    lines.append("")
    lines.append("  1. Shannon Entropy H(ε):")
    lines.append("     H = -Σ p_i log₂(p_i)")
    lines.append("     where p_i = N_i/N_total is the probability of finding")
    lines.append("     a particle in box i of size ε.")
    lines.append("     Units: bits")
    lines.append("")
    lines.append("  2. Lacunarity Λ(ε):")
    lines.append("     Λ = ⟨M²⟩/⟨M⟩² - 1")
    lines.append("     Measures heterogeneity/gappiness of the structure.")
    lines.append("     Λ = 0: uniform distribution")
    lines.append("     Λ > 0: heterogeneous (larger = more gaps)")
    lines.append("")
    lines.append("  3. Box-Counting Dimension D_0:")
    lines.append("     N(ε) ~ ε^(-D_0)")
    lines.append("     where N(ε) is the number of boxes of size ε")
    lines.append("     containing at least one particle.")
    lines.append("")
    lines.append("  4. Information Dimension D_1:")
    lines.append("     H(ε) ~ D_1 log(1/ε)")
    lines.append("     Characterizes how information content scales with resolution.")
    lines.append("")
    lines.append("  For monofractals: D_0 = D_1 = D_f (fractal dimension)")
    lines.append("  For multifractals: D_0 > D_1")
    lines.append("")
    
    # Summary table
    lines.append("=" * 75)
    lines.append("DIMENSION ESTIMATES SUMMARY")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"  {'Case':<25} {'D_f':>8} {'D_0':>8} {'D_1':>8} {'D_0-D_1':>8}")
    lines.append("  " + "-" * 60)
    
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        fits = all_fits[case_idx]
        
        D_f = data['fractal_dimension']
        D_0 = fits['D0']['D0'] if fits['D0'] else np.nan
        D_1 = fits['D1']['D1'] if fits['D1'] else np.nan
        diff = D_0 - D_1 if not (np.isnan(D_0) or np.isnan(D_1)) else np.nan
        
        lines.append(f"  {case_name:<25} {D_f:>8.4f} {D_0:>8.4f} {D_1:>8.4f} {diff:>8.4f}")
    lines.append("")
    lines.append("  D_f: Fractal dimension from mass-radius scaling")
    lines.append("  D_0: Box-counting dimension")
    lines.append("  D_1: Information dimension")
    lines.append("")
    
    # Detailed per-case analysis
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        results = all_results[case_idx]
        fits = all_fits[case_idx]
        
        lines.append("=" * 75)
        lines.append(f"CASE {case_idx + 1}: {case_name.upper()}")
        lines.append(f"File: {filename}")
        lines.append("=" * 75)
        lines.append("")
        
        # Basic properties
        lines.append("  AGGREGATE PROPERTIES:")
        lines.append(f"    Total particles:          {data['n_particles']:,}")
        lines.append(f"    Lattice size:             {data['lattice_size']} × {data['lattice_size']}")
        lines.append(f"    Fractal dimension D_f:    {data['fractal_dimension']:.4f}")
        lines.append("")
        
        # Box size range
        lines.append("  MULTISCALE ANALYSIS:")
        lines.append(f"    Box size range:           [{results['box_sizes'].min()}, {results['box_sizes'].max()}]")
        lines.append(f"    Number of scales:         {len(results['box_sizes'])}")
        lines.append("")
        
        # Entropy statistics
        lines.append("  SHANNON ENTROPY H(ε):")
        lines.append(f"    Minimum H:                {results['entropies'].min():.4f} bits")
        lines.append(f"    Maximum H:                {results['entropies'].max():.4f} bits")
        lines.append(f"    H at ε=2:                 {results['entropies'][0]:.4f} bits")
        lines.append(f"    H at ε=max:               {results['entropies'][-1]:.4f} bits")
        lines.append("")
        
        # Lacunarity statistics
        lines.append("  LACUNARITY Λ(ε):")
        lines.append(f"    Minimum Λ:                {results['lacunarities'].min():.4f}")
        lines.append(f"    Maximum Λ:                {results['lacunarities'].max():.4f}")
        lines.append(f"    Λ at ε=2:                 {results['lacunarities'][0]:.4f}")
        lines.append(f"    Λ at ε=max:               {results['lacunarities'][-1]:.4f}")
        lines.append("")
        
        # Box-counting dimension
        if fits['D0']:
            lines.append("  BOX-COUNTING DIMENSION D_0:")
            lines.append(f"    D_0:                      {fits['D0']['D0']:.4f} ± {fits['D0']['D0_err']:.4f}")
            lines.append(f"    R²:                       {fits['D0']['R2']:.4f}")
            lines.append(f"    p-value:                  {fits['D0']['p_value']:.2e}")
            
            # Compare with fractal dimension
            deviation = abs(fits['D0']['D0'] - data['fractal_dimension'])
            lines.append(f"    |D_0 - D_f|:              {deviation:.4f}")
            if deviation < 0.05:
                lines.append("    → Excellent agreement with mass-radius dimension")
            elif deviation < 0.10:
                lines.append("    → Good agreement with mass-radius dimension")
            else:
                lines.append("    → Some deviation from mass-radius dimension")
            lines.append("")
        
        # Information dimension
        if fits['D1']:
            lines.append("  INFORMATION DIMENSION D_1:")
            lines.append(f"    D_1:                      {fits['D1']['D1']:.4f} ± {fits['D1']['D1_err']:.4f}")
            lines.append(f"    R²:                       {fits['D1']['R2']:.4f}")
            lines.append(f"    p-value:                  {fits['D1']['p_value']:.2e}")
            
            if fits['D0']:
                diff = fits['D0']['D0'] - fits['D1']['D1']
                lines.append(f"    D_0 - D_1:                {diff:.4f}")
                if abs(diff) < 0.05:
                    lines.append("    → Consistent with monofractal structure")
                else:
                    lines.append("    → May indicate multifractal characteristics")
            lines.append("")
    
    # Comparative analysis
    lines.append("=" * 75)
    lines.append("COMPARATIVE ANALYSIS")
    lines.append("=" * 75)
    lines.append("")
    
    # Collect dimensions
    D_0_values = [f['D0']['D0'] for f in all_fits if f['D0']]
    D_1_values = [f['D1']['D1'] for f in all_fits if f['D1']]
    D_f_values = [d['fractal_dimension'] for d in all_data]
    
    lines.append("  DIMENSION STATISTICS:")
    lines.append(f"    Mean D_f:                 {np.mean(D_f_values):.4f} ± {np.std(D_f_values):.4f}")
    if D_0_values:
        lines.append(f"    Mean D_0:                 {np.mean(D_0_values):.4f} ± {np.std(D_0_values):.4f}")
    if D_1_values:
        lines.append(f"    Mean D_1:                 {np.mean(D_1_values):.4f} ± {np.std(D_1_values):.4f}")
    lines.append("")
    
    # Lacunarity comparison
    lines.append("  LACUNARITY COMPARISON (at ε = 4):")
    for case_idx, (filename, case_name) in enumerate(CASES):
        results = all_results[case_idx]
        idx = np.argmin(np.abs(results['box_sizes'] - 4))
        L = results['lacunarities'][idx]
        lines.append(f"    {case_name}: Λ = {L:.4f}")
    lines.append("")
    
    # Statistical tests
    lines.append("  KRUSKAL-WALLIS TEST (lacunarity curves):")
    all_lac = [r['lacunarities'] for r in all_results]
    # Use matched lengths
    min_len = min(len(L) for L in all_lac)
    all_lac_matched = [L[:min_len] for L in all_lac]
    
    stat_kw, p_kw = stats.kruskal(*all_lac_matched)
    sig = "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else ""
    lines.append(f"    H-statistic:              {stat_kw:.4f}")
    lines.append(f"    p-value:                  {p_kw:.4f} {sig}")
    if p_kw < 0.05:
        lines.append("    → Significant differences in lacunarity between cases")
    else:
        lines.append("    → No significant differences in lacunarity")
    lines.append("")
    
    # Interpretation
    lines.append("=" * 75)
    lines.append("INTERPRETATION")
    lines.append("=" * 75)
    lines.append("")
    lines.append("  1. ENTROPY SCALING:")
    lines.append("     The linear relationship between H(ε) and log(1/ε)")
    lines.append("     confirms self-similar scaling across multiple scales.")
    lines.append("     The slope D_1 (information dimension) characterizes")
    lines.append("     how the 'information content' of the structure scales.")
    lines.append("")
    lines.append("  2. LACUNARITY BEHAVIOR:")
    lines.append("     - High Λ at small ε: Structure appears gappy at fine scales")
    lines.append("     - Decreasing Λ with ε: Heterogeneity averages out at larger scales")
    lines.append("     - This scale-dependent behavior is characteristic of fractals")
    lines.append("")
    lines.append("  3. DIMENSION CONSISTENCY:")
    lines.append("     For ideal DLA (monofractal): D_f ≈ D_0 ≈ D_1 ≈ 1.71")
    lines.append("     Deviations may indicate:")
    lines.append("     - Finite-size effects (near boundaries)")
    lines.append("     - Statistical fluctuations (limited particles)")
    lines.append("     - Weak multifractal characteristics")
    lines.append("")
    lines.append("  4. PHYSICAL SIGNIFICANCE:")
    lines.append("     The information-theoretic measures quantify the")
    lines.append("     'complexity' of the aggregate structure, providing")
    lines.append("     a complementary view to the mass-radius fractal dimension.")
    lines.append("")
    
    lines.append("=" * 75)
    lines.append("END OF REPORT")
    lines.append("=" * 75)
    
    return "\n".join(lines)


# =============================================================================
# Plotting Functions
# =============================================================================

def setup_figure_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'font.weight': 'bold',
        'axes.labelsize': 13,
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'text.usetex': False,
        'mathtext.fontset': 'dejavuserif',
    })


def create_figure(all_data, all_results, all_fits):
    """Create the complete 2×2 panel figure."""
    setup_figure_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    axes = axes.flatten()
    
    # Store for legend
    legend_lines = []
    legend_labels = []
    
    # Panel (a): Shannon Entropy vs box size
    ax = axes[0]
    for case_idx, (filename, case_name) in enumerate(CASES):
        results = all_results[case_idx]
        
        line, = ax.plot(results['box_sizes'], results['entropies'],
                       color=COLORS[case_idx], linewidth=LINEWIDTHS[case_idx],
                       marker=MARKERS[case_idx], markersize=MARKERSIZE,
                       markerfacecolor='white', markeredgewidth=1.5,
                       label=case_name)
        legend_lines.append(line)
        legend_labels.append(case_name)
    
    ax.set_xlabel('Box size ε', fontsize=12, fontweight='bold')
    ax.set_ylabel('H(ε) [bits]', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.text(0.50, 1.05, PANEL_LABELS[0], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='center')
    
    # Panel (b): Lacunarity vs box size
    ax = axes[1]
    for case_idx, (filename, case_name) in enumerate(CASES):
        results = all_results[case_idx]
        
        ax.plot(results['box_sizes'], results['lacunarities'],
               color=COLORS[case_idx], linewidth=LINEWIDTHS[case_idx],
               marker=MARKERS[case_idx], markersize=MARKERSIZE,
               markerfacecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('Box size ε', fontsize=12, fontweight='bold')
    ax.set_ylabel('Λ(ε)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.text(0.50, 1.05, PANEL_LABELS[1], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='center')
    
    # Panel (c): Box-counting dimension
    ax = axes[2]
    for case_idx, (filename, case_name) in enumerate(CASES):
        results = all_results[case_idx]
        fits = all_fits[case_idx]
        
        log_eps = np.log(results['box_sizes'])
        log_N = np.log(results['n_occupied'])
        
        ax.scatter(log_eps, log_N, c=COLORS[case_idx], s=40,
                  marker=MARKERS[case_idx], alpha=0.7, edgecolors='white')
        
        # Plot fit line
        if fits['D0']:
            eps_fine = np.linspace(log_eps.min(), log_eps.max(), 100)
            N_fit = -fits['D0']['D0'] * eps_fine + fits['D0']['intercept']
            ax.plot(eps_fine, N_fit, color=COLORS[case_idx], 
                   linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('log ε', fontsize=12, fontweight='bold')
    ax.set_ylabel('log N(ε)', fontsize=12, fontweight='bold')
    ax.text(0.50, 1.05, PANEL_LABELS[2], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='center')
    
    # Add D_0 values as annotation
    y_pos = 0.95
    for case_idx, (filename, case_name) in enumerate(CASES):
        fits = all_fits[case_idx]
        if fits['D0']:
            ax.text(0.98, y_pos - 0.08 * case_idx, 
                   f"D₀ = {fits['D0']['D0']:.2f}",
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   color=COLORS[case_idx], ha='right', va='top')
    
    # Panel (d): Information dimension
    ax = axes[3]
    for case_idx, (filename, case_name) in enumerate(CASES):
        results = all_results[case_idx]
        fits = all_fits[case_idx]
        
        log_inv_eps = np.log(1 / results['box_sizes'])
        H = results['entropies']
        
        ax.scatter(log_inv_eps, H, c=COLORS[case_idx], s=40,
                  marker=MARKERS[case_idx], alpha=0.7, edgecolors='white')
        
        # Plot fit line
        if fits['D1']:
            x_fine = np.linspace(log_inv_eps.min(), log_inv_eps.max(), 100)
            H_fit = fits['D1']['D1'] * x_fine + fits['D1']['intercept']
            ax.plot(x_fine, H_fit, color=COLORS[case_idx],
                   linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('log(1/ε)', fontsize=12, fontweight='bold')
    ax.set_ylabel('H(ε) [bits]', fontsize=12, fontweight='bold')
    ax.text(0.50, 1.05, PANEL_LABELS[3], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='center')
    
    # Add D_1 values as annotation
    y_pos = 0.95
    for case_idx, (filename, case_name) in enumerate(CASES):
        fits = all_fits[case_idx]
        if fits['D1']:
            ax.text(0.02, y_pos - 0.08 * case_idx,
                   f"D₁ = {fits['D1']['D1']:.2f}",
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   color=COLORS[case_idx], ha='left', va='top')
    
    # Bold tick labels for all panels
    for ax in axes:
        for label in ax.xaxis.get_ticklabels():
            label.set_fontweight('bold')
        for label in ax.yaxis.get_ticklabels():
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.98], h_pad=2.5, w_pad=2.0)
    
    # Add shared legend at bottom
    fig.legend(legend_lines, legend_labels, loc='lower center', ncol=4,
              fontsize=11, frameon=True, fancybox=False, edgecolor='black',
              framealpha=1.0, bbox_to_anchor=(0.5, 0.01),
              handlelength=3, handletextpad=0.8, columnspacing=2.0)
    
    # Make legend text bold
    legend = fig.legends[0]
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    return fig


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 65)
    print("Spatial Entropy Analysis")
    print("=" * 65)
    
    # Create output directories
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    print("\n[1/5] Loading NetCDF data...")
    all_data = []
    for filename, case_name in CASES:
        filepath = DATA_DIR / filename
        print(f"      Loading {filename}...")
        data = load_netcdf(filepath)
        all_data.append(data)
        print(f"      → N = {data['n_particles']:,}, D_f = {data['fractal_dimension']:.4f}")
    
    # Perform multiscale analysis
    print("\n[2/5] Computing multiscale entropy and lacunarity...")
    all_results = []
    for case_idx, (filename, case_name) in enumerate(CASES):
        print(f"      Analyzing {case_name}...")
        results = multiscale_analysis(all_data[case_idx]['grid'])
        all_results.append(results)
        print(f"      → {len(results['box_sizes'])} scales analyzed")
    
    # Fit dimensions
    print("\n[3/5] Fitting information dimensions...")
    all_fits = []
    for case_idx, (filename, case_name) in enumerate(CASES):
        results = all_results[case_idx]
        
        fit_D0 = fit_box_counting_dimension(results['box_sizes'], results['n_occupied'])
        fit_D1 = fit_information_dimension(results['box_sizes'], results['entropies'])
        
        all_fits.append({'D0': fit_D0, 'D1': fit_D1})
        
        if fit_D0 and fit_D1:
            print(f"      {case_name}: D_0 = {fit_D0['D0']:.4f}, D_1 = {fit_D1['D1']:.4f}")
    
    # Create figure
    print("\n[4/5] Creating figure...")
    fig = create_figure(all_data, all_results, all_fits)
    
    # Save figures
    print("\n[5/5] Saving outputs...")
    
    # PNG
    png_path = FIGS_DIR / "spatial_entropy.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {png_path}")
    
    # PDF
    pdf_path = FIGS_DIR / "spatial_entropy.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {pdf_path}")
    
    # EPS
    eps_path = FIGS_DIR / "spatial_entropy.eps"
    fig.savefig(eps_path, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {eps_path}")
    
    plt.close(fig)
    
    # Generate and save report
    print("\n      Generating statistical report...")
    report = compute_statistics(all_data, all_results, all_fits)
    
    report_path = REPORTS_DIR / "spatial_entropy_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"      Saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"\nFigures saved to: {FIGS_DIR}")
    print(f"Report saved to: {REPORTS_DIR}")
    print("\nPanel layout (2×2):")
    print("  (a) Shannon entropy H(ε) vs box size")
    print("  (b) Lacunarity Λ(ε) vs box size")
    print("  (c) Box-counting dimension D_0")
    print("  (d) Information dimension D_1")
    print("\nDone!")


if __name__ == "__main__":
    main()
