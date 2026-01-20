#!/usr/bin/env python
"""
Publication-quality growth dynamics analysis for DLA aggregates.

Creates a 2×2 panel figure:
  - (a) Cumulative particle count N(t) vs snapshot index (all cases)
  - (b) Instantaneous growth rate dN/dt vs N (all cases)
  - (c) Normalized growth curves N(t)/N_max (all cases)  
  - (d) Growth efficiency: particles per effective radius vs time

Growth dynamics in DLA:
  - Early growth: Rapid accumulation, tips easily accessible
  - Late growth: Screening reduces growth rate as fjords fill
  - Theoretical: N(t) ~ t^(D/2) for diffusion-controlled growth

Output:
  - ../figs/growth_dynamics.png (300 dpi)
  - ../figs/growth_dynamics.pdf (vector)
  - ../figs/growth_dynamics.eps (vector)
  - ../reports/growth_dynamics_report.txt

Author: Sandy H. S. Herho
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
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

# Colors for 4 cases - colorblind friendly
COLORS = [
    '#E63946',  # Red - Case 1
    '#457B9D',  # Blue - Case 2
    '#2A9D8F',  # Teal - Case 3
    '#F4A261',  # Orange - Case 4
]

LINEWIDTHS = [2.5, 2.5, 2.5, 2.5]
ALPHAS = [0.9, 0.9, 0.9, 0.9]


# =============================================================================
# Data Loading
# =============================================================================

def load_netcdf(filepath):
    """Load NetCDF data and extract key variables."""
    with Dataset(filepath, 'r') as nc:
        data = {
            'grid': nc.variables['grid'][:],
            'snapshots': nc.variables['snapshots'][:],
            'glued_counts': nc.variables['glued_counts'][:],
            'radii': nc.variables['radii'][:],
            'masses': nc.variables['masses'][:],
            'n_particles': nc.n_particles,
            'n_aggregates': nc.n_aggregates,
            'fractal_dimension': nc.fractal_dimension,
            'center_x': nc.center_x,
            'center_y': nc.center_y,
            'lattice_size': nc.lattice_size,
            'n_walkers': nc.n_walkers,
            'n_seeds': nc.n_seeds,
            'n_iterations': nc.n_iterations,
            'scenario': nc.scenario,
        }
    return data


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_growth_rate(counts, smoothing=True):
    """
    Compute instantaneous growth rate dN/dt.
    
    Args:
        counts: Cumulative particle counts
        smoothing: Apply Savitzky-Golay filter
    
    Returns:
        rates: Growth rate at each time point
    """
    if len(counts) < 3:
        return np.zeros(len(counts))
    
    # Central difference
    rates = np.gradient(counts)
    
    if smoothing and len(rates) > 11:
        # Savitzky-Golay filter for smoothing
        window = min(11, len(rates) // 2 * 2 - 1)
        if window >= 5:
            rates = savgol_filter(rates, window, 3)
    
    return rates


def compute_effective_radius(grid, center_x, center_y):
    """Compute effective radius (radius of gyration) of aggregate."""
    agg_mask = (grid == 2)
    if not np.any(agg_mask):
        return 0
    
    coords = np.argwhere(agg_mask)
    distances_sq = (coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2
    
    return np.sqrt(np.mean(distances_sq))


def analyze_growth_phases(counts):
    """
    Analyze growth phases: initial, linear, saturation.
    
    Returns:
        dict with phase boundaries and characteristics
    """
    if len(counts) < 10:
        return None
    
    n_max = counts[-1]
    
    # Find phases based on fraction of final mass
    initial_end = np.searchsorted(counts, 0.1 * n_max)
    saturation_start = np.searchsorted(counts, 0.9 * n_max)
    
    # Growth rates in each phase
    rates = compute_growth_rate(counts)
    
    result = {
        'n_snapshots': len(counts),
        'n_final': n_max,
        'initial_phase': (0, initial_end),
        'growth_phase': (initial_end, saturation_start),
        'saturation_phase': (saturation_start, len(counts) - 1),
    }
    
    # Statistics for each phase
    for phase_name, (start, end) in [('initial', result['initial_phase']),
                                      ('growth', result['growth_phase']),
                                      ('saturation', result['saturation_phase'])]:
        if end > start:
            phase_rates = rates[start:end]
            result[f'{phase_name}_mean_rate'] = np.mean(phase_rates)
            result[f'{phase_name}_std_rate'] = np.std(phase_rates)
        else:
            result[f'{phase_name}_mean_rate'] = np.nan
            result[f'{phase_name}_std_rate'] = np.nan
    
    return result


def fit_growth_exponent(counts):
    """
    Fit power-law growth: N(t) ~ t^alpha
    
    For diffusion-limited growth, alpha ≈ D/2 where D is fractal dimension.
    """
    if len(counts) < 10:
        return None
    
    # Use middle portion to avoid edge effects
    n = len(counts)
    start = n // 10
    end = 9 * n // 10
    
    t = np.arange(start, end) + 1  # Avoid log(0)
    N = counts[start:end]
    
    # Filter positive values
    valid = N > 0
    if np.sum(valid) < 5:
        return None
    
    log_t = np.log(t[valid])
    log_N = np.log(N[valid])
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_N)
    
    return {
        'alpha': slope,
        'alpha_err': std_err,
        'R2': r_value**2,
        'p_value': p_value,
        'prefactor': np.exp(intercept),
    }


# =============================================================================
# Statistics Report
# =============================================================================

def compute_statistics(all_data):
    """Compute comprehensive statistics and return formatted report."""
    lines = []
    lines.append("=" * 75)
    lines.append("GROWTH DYNAMICS ANALYSIS REPORT")
    lines.append("=" * 75)
    lines.append("")
    lines.append("Analysis of DLA aggregate growth curves and rates.")
    lines.append("")
    lines.append("Panel descriptions:")
    lines.append("  (a) Cumulative growth: N(t) vs snapshot index")
    lines.append("  (b) Growth rate: dN/dt vs particle count N")
    lines.append("  (c) Normalized growth: N(t)/N_max for shape comparison")
    lines.append("  (d) Growth efficiency: particles per unit radius")
    lines.append("")
    lines.append("Theoretical expectation: N(t) ~ t^α with α ≈ D/2 ≈ 0.855")
    lines.append("for 2D DLA where D ≈ 1.71")
    lines.append("")
    
    # Summary table
    lines.append("=" * 75)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"  {'Case':<25} {'N_final':>10} {'Snapshots':>10} {'Mean Rate':>12}")
    lines.append("  " + "-" * 60)
    
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        counts = data['glued_counts']
        rates = compute_growth_rate(counts)
        lines.append(f"  {case_name:<25} {counts[-1]:>10,} {len(counts):>10} "
                    f"{rates.mean():>12.2f}")
    lines.append("")
    
    # Detailed per-case analysis
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        counts = data['glued_counts']
        rates = compute_growth_rate(counts)
        
        lines.append("=" * 75)
        lines.append(f"CASE {case_idx + 1}: {case_name.upper()}")
        lines.append(f"File: {filename}")
        lines.append("=" * 75)
        lines.append("")
        
        # Basic growth statistics
        lines.append("  GROWTH CURVE STATISTICS:")
        lines.append(f"    Number of snapshots:      {len(counts)}")
        lines.append(f"    Initial particles:        {counts[0]:,}")
        lines.append(f"    Final particles:          {counts[-1]:,}")
        lines.append(f"    Net growth:               {counts[-1] - counts[0]:,}")
        lines.append(f"    Target walkers:           {data['n_walkers']:,}")
        lines.append(f"    Completion rate:          {100 * counts[-1] / data['n_walkers']:.1f}%")
        lines.append("")
        
        # Growth rate statistics
        lines.append("  GROWTH RATE STATISTICS (dN/dt):")
        lines.append(f"    Mean rate:                {rates.mean():.4f} particles/snapshot")
        lines.append(f"    Std rate:                 {rates.std():.4f}")
        lines.append(f"    Max rate:                 {rates.max():.4f}")
        lines.append(f"    Min rate:                 {rates.min():.4f}")
        lines.append(f"    CV (Std/Mean):            {100 * rates.std() / rates.mean():.1f}%")
        lines.append("")
        
        # Growth phases
        phases = analyze_growth_phases(counts)
        if phases:
            lines.append("  GROWTH PHASES:")
            lines.append(f"    Initial phase (0-10%):    snapshots {phases['initial_phase'][0]}-{phases['initial_phase'][1]}")
            lines.append(f"      Mean rate:              {phases['initial_mean_rate']:.4f}")
            lines.append(f"    Growth phase (10-90%):    snapshots {phases['growth_phase'][0]}-{phases['growth_phase'][1]}")
            lines.append(f"      Mean rate:              {phases['growth_mean_rate']:.4f}")
            lines.append(f"    Saturation phase (90%+):  snapshots {phases['saturation_phase'][0]}-{phases['saturation_phase'][1]}")
            lines.append(f"      Mean rate:              {phases['saturation_mean_rate']:.4f}")
            lines.append("")
        
        # Power-law fit
        fit = fit_growth_exponent(counts)
        if fit:
            lines.append("  POWER-LAW FIT: N(t) ~ t^α")
            lines.append(f"    Growth exponent α:        {fit['alpha']:.4f} ± {fit['alpha_err']:.4f}")
            lines.append(f"    Expected (D/2 ≈ 0.855):   {data['fractal_dimension']/2:.4f}")
            lines.append(f"    R² (goodness of fit):     {fit['R2']:.4f}")
            lines.append(f"    p-value:                  {fit['p_value']:.2e}")
            lines.append("")
        
        # Temporal correlation analysis
        lines.append("  TEMPORAL CORRELATION:")
        if len(rates) > 10:
            # Autocorrelation of growth rates
            rates_centered = rates - rates.mean()
            autocorr = np.correlate(rates_centered, rates_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
            
            # Find correlation length (first zero crossing or 1/e)
            try:
                corr_length = np.where(autocorr < 1/np.e)[0][0]
            except IndexError:
                corr_length = len(autocorr)
            
            lines.append(f"    Autocorrelation at lag 1: {autocorr[1]:.4f}")
            lines.append(f"    Correlation length:       {corr_length} snapshots")
            
            # Ljung-Box test for white noise
            if len(rates) >= 20:
                from scipy.stats import chi2
                n = len(rates)
                k = min(10, n // 4)
                Q = n * (n + 2) * sum((autocorr[i]**2) / (n - i) for i in range(1, k + 1))
                p_lb = 1 - chi2.cdf(Q, k)
                lines.append(f"    Ljung-Box Q (lag {k}):     {Q:.4f}")
                lines.append(f"    p-value (white noise):    {p_lb:.4f}")
                if p_lb > 0.05:
                    lines.append("    → Growth rate consistent with random process")
                else:
                    lines.append("    → Significant temporal structure in growth rate")
        lines.append("")
    
    # Comparative analysis
    lines.append("=" * 75)
    lines.append("COMPARATIVE ANALYSIS")
    lines.append("=" * 75)
    lines.append("")
    
    # Growth efficiency comparison
    lines.append("  GROWTH EFFICIENCY (final N / target N):")
    efficiencies = []
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        eff = data['n_particles'] / data['n_walkers']
        efficiencies.append((case_name, eff))
    
    efficiencies.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, eff) in enumerate(efficiencies, 1):
        lines.append(f"    {rank}. {name}: {100*eff:.1f}%")
    lines.append("")
    
    # Rate variability comparison
    lines.append("  GROWTH RATE VARIABILITY (CV = Std/Mean):")
    cvs = []
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        rates = compute_growth_rate(data['glued_counts'])
        cv = rates.std() / rates.mean() if rates.mean() > 0 else np.inf
        cvs.append((case_name, cv))
    
    cvs.sort(key=lambda x: x[1])
    for rank, (name, cv) in enumerate(cvs, 1):
        lines.append(f"    {rank}. {name}: CV = {100*cv:.1f}%")
    lines.append("")
    
    # Statistical tests between cases
    lines.append("  KRUSKAL-WALLIS TEST (growth rates across cases):")
    all_rates = [compute_growth_rate(all_data[i]['glued_counts']) for i in range(len(CASES))]
    stat_kw, p_kw = stats.kruskal(*all_rates)
    sig = "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else ""
    lines.append(f"    H-statistic:              {stat_kw:.4f}")
    lines.append(f"    p-value:                  {p_kw:.2e} {sig}")
    if p_kw < 0.05:
        lines.append("    → Significant differences in growth rates between cases")
    else:
        lines.append("    → No significant differences in growth rates")
    lines.append("")
    
    # Interpretation
    lines.append("=" * 75)
    lines.append("INTERPRETATION")
    lines.append("=" * 75)
    lines.append("")
    lines.append("  1. GROWTH CURVE SHAPE:")
    lines.append("     All cases show characteristic DLA growth: initial rapid")
    lines.append("     accumulation followed by gradual slowdown as screening")
    lines.append("     effects become more pronounced with increasing aggregate size.")
    lines.append("")
    lines.append("  2. SCREENING EFFECT:")
    lines.append("     Decreasing growth rate with N reflects diffusive screening:")
    lines.append("     larger aggregates present more surface area but also more")
    lines.append("     screening of interior sites by protruding tips.")
    lines.append("")
    lines.append("  3. INJECTION MODE EFFECTS:")
    lines.append("     - Random injection: Growth rate depends on walker distribution")
    lines.append("     - Radial injection: More uniform flux leads to steadier growth")
    lines.append("     - Multiple seeds: Competition initially, then coalescence effects")
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


def create_figure(all_data):
    """Create the complete 2×2 panel figure."""
    setup_figure_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    axes = axes.flatten()
    
    # Store line objects for legend
    legend_lines = []
    legend_labels = []
    
    # Panel (a): Cumulative growth N(t)
    ax = axes[0]
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        counts = data['glued_counts']
        t = np.arange(len(counts))
        
        line, = ax.plot(t, counts, color=COLORS[case_idx], 
                       linewidth=LINEWIDTHS[case_idx], alpha=ALPHAS[case_idx],
                       label=case_name)
        if case_idx == 0:
            legend_lines.append(line)
            legend_labels.append(case_name)
        else:
            legend_lines.append(line)
            legend_labels.append(case_name)
    
    ax.set_xlabel('Snapshot Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('N(t)', fontsize=12, fontweight='bold')
    ax.text(0.50, 1.05, PANEL_LABELS[0], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='center')
    
    # Panel (b): Growth rate dN/dt vs N
    ax = axes[1]
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        counts = data['glued_counts']
        rates = compute_growth_rate(counts)
        
        ax.plot(counts, rates, color=COLORS[case_idx],
               linewidth=LINEWIDTHS[case_idx], alpha=ALPHAS[case_idx])
    
    ax.set_xlabel('N', fontsize=12, fontweight='bold')
    ax.set_ylabel('dN/dt', fontsize=12, fontweight='bold')
    ax.text(0.50, 1.05, PANEL_LABELS[1], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='center')
    
    # Panel (c): Normalized growth N(t)/N_max
    ax = axes[2]
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        counts = data['glued_counts']
        n_max = counts[-1]
        t_norm = np.linspace(0, 1, len(counts))
        
        ax.plot(t_norm, counts / n_max, color=COLORS[case_idx],
               linewidth=LINEWIDTHS[case_idx], alpha=ALPHAS[case_idx])
    
    ax.set_xlabel('t / T', fontsize=12, fontweight='bold')
    ax.set_ylabel('N(t) / N$_{max}$', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.text(0.50, 1.05, PANEL_LABELS[2], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='center')
    
    # Panel (d): Growth efficiency - N/R_eff vs time
    ax = axes[3]
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        counts = data['glued_counts']
        
        # Estimate effective radius from particle count using D
        D = data['fractal_dimension']
        R_eff = np.power(counts, 1/D)  # R ~ N^(1/D)
        efficiency = counts / np.maximum(R_eff, 1)
        
        t_norm = np.linspace(0, 1, len(counts))
        ax.plot(t_norm, efficiency / efficiency.max(), color=COLORS[case_idx],
               linewidth=LINEWIDTHS[case_idx], alpha=ALPHAS[case_idx])
    
    ax.set_xlabel('t / T', fontsize=12, fontweight='bold')
    ax.set_ylabel('N / R$_{eff}$ (normalized)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.text(0.50, 1.05, PANEL_LABELS[3], transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='bottom', ha='center')
    
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
    print("Growth Dynamics Analysis")
    print("=" * 65)
    
    # Create output directories
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    print("\n[1/4] Loading NetCDF data...")
    all_data = []
    for filename, case_name in CASES:
        filepath = DATA_DIR / filename
        print(f"      Loading {filename}...")
        data = load_netcdf(filepath)
        all_data.append(data)
        print(f"      → {len(data['glued_counts'])} snapshots, "
              f"final N = {data['n_particles']:,}")
    
    # Create figure
    print("\n[2/4] Creating figure...")
    fig = create_figure(all_data)
    
    # Save figures
    print("\n[3/4] Saving figures...")
    
    # PNG
    png_path = FIGS_DIR / "growth_dynamics.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {png_path}")
    
    # PDF
    pdf_path = FIGS_DIR / "growth_dynamics.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {pdf_path}")
    
    # EPS
    eps_path = FIGS_DIR / "growth_dynamics.eps"
    fig.savefig(eps_path, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {eps_path}")
    
    plt.close(fig)
    
    # Generate and save report
    print("\n[4/4] Generating statistical report...")
    report = compute_statistics(all_data)
    
    report_path = REPORTS_DIR / "growth_dynamics_report.txt"
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
    print("  (a) Cumulative growth N(t)")
    print("  (b) Growth rate dN/dt vs N")
    print("  (c) Normalized growth curves")
    print("  (d) Growth efficiency N/R_eff")
    print("\nDone!")


if __name__ == "__main__":
    main()
