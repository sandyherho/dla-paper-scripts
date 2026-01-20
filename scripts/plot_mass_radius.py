#!/usr/bin/env python
"""
Publication-quality mass-radius scaling analysis for fractal dimension.
1×3 layout - excludes Multiple Seeds (invalid for mass-radius analysis).

Multiple Seeds case excluded because mass-radius D requires a single connected
aggregate. Disconnected clusters artificially inflate D.

Creates a 1×3 panel figure with shared legend at bottom:
  (a) Classic DLA
  (b) Radial Injection  
  (c) High Density

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

# Only single-aggregate cases (skip Multiple Seeds)
CASES = [
    ("case_1_classic_dla.nc", "Classic DLA"),
    ("case_3_radial_injection.nc", "Radial Injection"),
    ("case_4_high_density.nc", "High Density"),
]

FIG_WIDTH, FIG_HEIGHT, DPI = 14, 5, 300
PANEL_LABELS = ["(a)", "(b)", "(c)"]

DATA_COLOR = '#264653'
FIT_COLOR = '#E63946'
BAND_COLOR = '#E63946'
THEORY_COLOR = '#2A9D8F'

D_THEORY = 1.71
D_THEORY_ERR = 0.01


def load_netcdf(filepath):
    with Dataset(filepath, 'r') as nc:
        return {
            'grid': nc.variables['grid'][:],
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
            'scenario': nc.scenario,
        }


def fit_fractal_dimension(radii, masses, confidence=0.95):
    """Fit D with bootstrap confidence intervals."""
    valid = (radii > 0) & (masses > 0) & np.isfinite(radii) & np.isfinite(masses)
    r, m = radii[valid], masses[valid]
    if len(r) < 5:
        return None
    
    log_r, log_m = np.log(r), np.log(m)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_m)
    
    residuals = log_m - (slope * log_r + intercept)
    
    # Bootstrap
    n_bootstrap, n_points = 1000, len(log_r)
    D_bootstrap = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.choice(n_points, n_points, replace=True)
        D_bootstrap[i] = stats.linregress(log_r[idx], log_m[idx])[0]
    
    alpha = 1 - confidence
    D_err = (np.percentile(D_bootstrap, 100*(1-alpha/2)) - np.percentile(D_bootstrap, 100*alpha/2)) / 2
    
    return {
        'D': slope, 'D_err': D_err, 'D_std_err': std_err,
        'D_ci_low': np.percentile(D_bootstrap, 100*alpha/2),
        'D_ci_high': np.percentile(D_bootstrap, 100*(1-alpha/2)),
        'intercept': intercept, 'R2': r_value**2, 'p_value': p_value,
        'residuals': residuals, 'log_r': log_r, 'log_m': log_m, 'r': r, 'm': m,
        'n_points': len(r), 'rmse': np.sqrt(np.mean(residuals**2)),
    }


def compute_prediction_band(log_r, slope, intercept, residuals, confidence=0.95):
    n, dof = len(log_r), len(log_r) - 2
    log_r_mean = np.mean(log_r)
    ss_x = np.sum((log_r - log_r_mean)**2)
    mse = np.sum(residuals**2) / dof
    t_val = stats.t.ppf(1 - (1-confidence)/2, dof)
    
    r_fine = np.linspace(log_r.min(), log_r.max(), 100)
    y_pred = slope * r_fine + intercept
    se_conf = np.sqrt(mse * (1/n + (r_fine - log_r_mean)**2 / ss_x))
    
    return r_fine, y_pred, y_pred - t_val*se_conf, y_pred + t_val*se_conf


def compute_statistics(all_data, fit_results):
    lines = ["=" * 75, "MASS-RADIUS SCALING ANALYSIS REPORT", "=" * 75, "",
             "Fractal dimension via mass-radius scaling: M(R) ~ R^D", "",
             "NOTE: Multiple Seeds case excluded (disconnected clusters invalid", 
             "      for mass-radius analysis). Use box-counting D_0 for that case.", "",
             f"Theoretical 2D DLA: D = {D_THEORY} +/- {D_THEORY_ERR}", ""]
    
    lines.extend(["=" * 75, "SUMMARY TABLE", "=" * 75, "",
                  f"  {'Case':<25} {'D':>10} {'+/-err':>8} {'R^2':>8} {'|D-1.71|':>10}",
                  "  " + "-" * 64])
    
    for i, (fn, name) in enumerate(CASES):
        f = fit_results[i]
        if f:
            dev = abs(f['D'] - D_THEORY)
            lines.append(f"  {name:<25} {f['D']:>10.4f} {f['D_err']:>8.4f} {f['R2']:>8.4f} {dev:>10.4f}")
    
    lines.append("")
    
    # Statistics
    Ds = [f['D'] for f in fit_results if f]
    lines.extend([f"  Mean D:                   {np.mean(Ds):.4f} +/- {np.std(Ds):.4f}",
                  f"  Theoretical D:            {D_THEORY}",
                  f"  Mean deviation:           {np.mean([abs(d - D_THEORY) for d in Ds]):.4f}", ""])
    
    # Per-case details
    for i, (fn, name) in enumerate(CASES):
        d, f = all_data[i], fit_results[i]
        lines.extend(["=" * 75, f"CASE: {name.upper()}", f"File: {fn}", "=" * 75, ""])
        
        lines.extend(["  DATA PROPERTIES:",
                      f"    Total particles:          {d['n_particles']:,}",
                      f"    Lattice size:             {d['lattice_size']} x {d['lattice_size']}",
                      f"    Stored D (from solver):   {d['fractal_dimension']:.4f}", ""])
        
        if f:
            lines.extend(["  LINEAR REGRESSION (log-log space):",
                          f"    Number of data points:    {f['n_points']}",
                          f"    Radius range:             [{f['r'].min():.2f}, {f['r'].max():.2f}]",
                          f"    Mass range:               [{int(f['m'].min())}, {int(f['m'].max())}]", "",
                          f"    Slope (D):                {f['D']:.6f}",
                          f"    Standard error:           {f['D_std_err']:.6f}",
                          f"    95% CI:                   [{f['D_ci_low']:.4f}, {f['D_ci_high']:.4f}]",
                          f"    Bootstrap error:          +/-{f['D_err']:.4f}", "",
                          f"    Intercept (log c):        {f['intercept']:.6f}", ""])
            
            lines.extend(["  GOODNESS OF FIT:",
                          f"    R^2 (coefficient):        {f['R2']:.6f}",
                          f"    p-value (slope != 0):     {f['p_value']:.2e}",
                          f"    RMSE (log space):         {f['rmse']:.6f}", ""])
            
            dev = abs(f['D'] - D_THEORY)
            z_score = (f['D'] - D_THEORY) / f['D_std_err']
            p_theory = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            lines.extend(["  COMPARISON WITH THEORY (D = 1.71):",
                          f"    Deviation:                {f['D'] - D_THEORY:+.4f}",
                          f"    Relative deviation:       {100*(f['D'] - D_THEORY)/D_THEORY:+.2f}%",
                          f"    z-score:                  {z_score:.2f}",
                          f"    p-value (D = 1.71):       {p_theory:.4f}"])
            
            if p_theory > 0.05:
                lines.append("    -> Consistent with theoretical value (p > 0.05)")
            else:
                lines.append("    -> Statistically different from theory (p < 0.05)")
            lines.append("")
    
    # Interpretation
    lines.extend(["=" * 75, "INTERPRETATION", "=" * 75, "",
                  "  The fractal dimension D characterizes mass-radius scaling in DLA.",
                  "  For 2D DLA, theory predicts D = 1.71, between a line (D=1) and",
                  "  a filled disk (D=2), reflecting the ramified dendritic morphology.", "",
                  "  Results:"])
    
    for i, (_, name) in enumerate(CASES):
        f = fit_results[i]
        if f:
            dev = abs(f['D'] - D_THEORY)
            qual = "excellent" if dev < 0.02 else "good" if dev < 0.1 else "moderate"
            lines.append(f"    - {name}: D = {f['D']:.3f} ({qual} agreement)")
    
    lines.extend(["", "  Note: Multiple Seeds case excluded from this analysis.",
                  "  For spatially distributed clusters, use box-counting dimension.", "",
                  "=" * 75, "END OF REPORT", "=" * 75])
    
    return "\n".join(lines)


def setup_figure_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11, 'font.weight': 'bold',
        'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.linewidth': 1.5,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.8,
    })


def plot_mass_radius_panel(ax, data, fit_result, panel_label):
    """Plot mass-radius data with fit for a single case."""
    if fit_result is None:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
        return
    
    log_r, log_m = fit_result['log_r'], fit_result['log_m']
    D, D_err, R2 = fit_result['D'], fit_result['D_err'], fit_result['R2']
    
    # Data points
    ax.scatter(log_r, log_m, c=DATA_COLOR, s=40, alpha=0.7, 
               edgecolors='white', linewidths=0.5, zorder=3, label='Data')
    
    # Confidence band and fit
    r_fine, y_pred, y_lower, y_upper = compute_prediction_band(
        log_r, D, fit_result['intercept'], fit_result['residuals'])
    
    ax.fill_between(r_fine, y_lower, y_upper, color=BAND_COLOR, alpha=0.15, 
                    zorder=1, label='95% CI')
    ax.plot(r_fine, y_pred, color=FIT_COLOR, linewidth=2.5, zorder=2,
            label=f'Fit: D = {D:.3f}+/-{D_err:.3f}')
    
    # Theoretical slope
    y_theory = D_THEORY * r_fine + (fit_result['intercept'] + (D - D_THEORY) * np.mean(log_r))
    ax.plot(r_fine, y_theory, color=THEORY_COLOR, linewidth=1.5, 
            linestyle='--', alpha=0.7, zorder=2, label=f'Theory: D = {D_THEORY}')
    
    ax.set_xlabel(r'log R', fontweight='bold')
    ax.set_ylabel(r'log M(R)', fontweight='bold')
    
    # Panel label
    ax.text(0.50, 1.05, panel_label, transform=ax.transAxes, fontsize=16, 
            fontweight='bold', va='bottom', ha='center')
    
    # R^2 box
    ax.text(0.05, 0.95, f'R$^2$ = {R2:.4f}', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    for label in ax.xaxis.get_ticklabels():
        label.set_fontweight('bold')
    for label in ax.yaxis.get_ticklabels():
        label.set_fontweight('bold')


def create_figure(all_data, fit_results):
    """Create 1x3 panel figure with shared legend at bottom."""
    setup_figure_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    
    for i, (fn, name) in enumerate(CASES):
        plot_mass_radius_panel(axes[i], all_data[i], fit_results[i], PANEL_LABELS[i])
    
    plt.tight_layout(rect=[0, 0.14, 1, 0.98], w_pad=2.5)
    
    # Shared legend at bottom
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=DATA_COLOR, 
                   markersize=8, label='Data'),
        plt.Rectangle((0,0), 1, 1, fc=BAND_COLOR, alpha=0.15, label='95% CI'),
        plt.Line2D([0], [0], color=FIT_COLOR, linewidth=2.5, label='Fit'),
        plt.Line2D([0], [0], color=THEORY_COLOR, linewidth=1.5, linestyle='--', 
                   label=f'Theory: D = {D_THEORY}'),
    ]
    
    leg = fig.legend(handles=legend_handles, loc='lower center', ncol=4,
                    fontsize=11, frameon=True, fancybox=False, edgecolor='black',
                    framealpha=1.0, bbox_to_anchor=(0.5, 0.01),
                    handlelength=2.5, handletextpad=0.8, columnspacing=2.5)
    
    for text in leg.get_texts():
        text.set_fontweight('bold')
    
    return fig


def main():
    print("=" * 65)
    print("Mass-Radius Scaling Analysis")
    print("=" * 65)
    print("\nNote: Multiple Seeds case excluded (invalid for mass-radius D)")
    print("      Analyzing 3 single-aggregate cases\n")
    
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[1/4] Loading data...")
    all_data = []
    for fn, name in CASES:
        filepath = DATA_DIR / fn
        print(f"      Loading {fn}...")
        data = load_netcdf(filepath)
        all_data.append(data)
    
    print("\n[2/4] Fitting fractal dimensions...")
    fit_results = [fit_fractal_dimension(d['radii'], d['masses']) for d in all_data]
    
    for i, (_, name) in enumerate(CASES):
        f = fit_results[i]
        if f:
            dev = abs(f['D'] - D_THEORY)
            print(f"      {name}: D = {f['D']:.4f} +/- {f['D_err']:.4f} (delta = {dev:.4f})")
    
    print("\n[3/4] Creating figure...")
    fig = create_figure(all_data, fit_results)
    
    png_path = FIGS_DIR / "mass_radius_scaling.png"
    pdf_path = FIGS_DIR / "mass_radius_scaling.pdf"
    
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      Saved: {png_path}")
    print(f"      Saved: {pdf_path}")
    
    print("\n[4/4] Generating report...")
    report = compute_statistics(all_data, fit_results)
    report_path = REPORTS_DIR / "mass_radius_scaling_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"      Saved: {report_path}")
    
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"\n{'Case':<25} {'D':>10} {'|D-1.71|':>10}")
    print("-" * 47)
    for i, (_, name) in enumerate(CASES):
        f = fit_results[i]
        if f:
            print(f"{name:<25} {f['D']:>10.4f} {abs(f['D']-D_THEORY):>10.4f}")
    print(f"\nTheoretical: D = {D_THEORY}")
    print("\nDone!")


if __name__ == "__main__":
    main()
