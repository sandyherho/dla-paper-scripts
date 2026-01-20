#!/usr/bin/env python
"""
Publication-quality mass-radius scaling analysis for fractal dimension.

Creates a 2×2 panel figure:
  - Each panel shows log-log M(R) vs R for one case
  - Linear regression fit with confidence band
  - Extracted fractal dimension with uncertainty

The fractal dimension D is obtained from:
    M(R) ~ R^D  →  log M = D log R + const

Theoretical value for 2D DLA: D ≈ 1.71 ± 0.01

Output:
  - ../figs/mass_radius_scaling.png (300 dpi)
  - ../figs/mass_radius_scaling.pdf (vector)
  - ../figs/mass_radius_scaling.eps (vector)
  - ../reports/mass_radius_scaling_report.txt

Author: Sandy H. S. Herho
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
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
FIG_WIDTH = 10
FIG_HEIGHT = 10
DPI = 300

# Panel labels
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

# Colors
DATA_COLOR = '#264653'      # Dark blue for data points
FIT_COLOR = '#E63946'       # Red for fit line
BAND_COLOR = '#E63946'      # Same for confidence band
THEORY_COLOR = '#2A9D8F'    # Teal for theoretical slope

# Theoretical fractal dimension
D_THEORY = 1.71
D_THEORY_ERR = 0.01


# =============================================================================
# Data Loading
# =============================================================================

def load_netcdf(filepath):
    """Load NetCDF data and extract key variables."""
    with Dataset(filepath, 'r') as nc:
        data = {
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
    return data


# =============================================================================
# Analysis Functions
# =============================================================================

def power_law(x, D, c):
    """Power law function M = c * R^D."""
    return c * np.power(x, D)


def linear_model(x, slope, intercept):
    """Linear model for log-log regression."""
    return slope * x + intercept


def fit_fractal_dimension(radii, masses, confidence=0.95):
    """
    Fit fractal dimension with uncertainty estimation.
    
    Uses log-log linear regression with bootstrap for confidence intervals.
    
    Returns:
        dict with D, D_err, R2, residuals, fit parameters
    """
    # Filter valid data
    valid = (radii > 0) & (masses > 0) & np.isfinite(radii) & np.isfinite(masses)
    r = radii[valid]
    m = masses[valid]
    
    if len(r) < 5:
        return None
    
    # Log transform
    log_r = np.log(r)
    log_m = np.log(m)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_m)
    
    # Residuals
    predicted = slope * log_r + intercept
    residuals = log_m - predicted
    
    # Bootstrap for confidence interval
    n_bootstrap = 1000
    D_bootstrap = np.zeros(n_bootstrap)
    
    n_points = len(log_r)
    for i in range(n_bootstrap):
        idx = np.random.choice(n_points, n_points, replace=True)
        s, _, _, _, _ = stats.linregress(log_r[idx], log_m[idx])
        D_bootstrap[i] = s
    
    # Confidence interval
    alpha = 1 - confidence
    D_low = np.percentile(D_bootstrap, 100 * alpha / 2)
    D_high = np.percentile(D_bootstrap, 100 * (1 - alpha / 2))
    D_err = (D_high - D_low) / 2
    
    # Additional statistics
    n = len(log_r)
    dof = n - 2
    t_stat = stats.t.ppf(1 - alpha / 2, dof)
    
    # Standard error of estimate
    mse = np.sum(residuals**2) / dof
    se_slope = std_err
    
    return {
        'D': slope,
        'D_err': D_err,
        'D_std_err': std_err,
        'D_ci_low': D_low,
        'D_ci_high': D_high,
        'intercept': intercept,
        'R2': r_value**2,
        'p_value': p_value,
        'residuals': residuals,
        'log_r': log_r,
        'log_m': log_m,
        'r': r,
        'm': m,
        'n_points': n,
        'mse': mse,
        'rmse': np.sqrt(mse),
    }


def compute_prediction_band(log_r, slope, intercept, residuals, confidence=0.95):
    """Compute prediction band for linear fit."""
    n = len(log_r)
    dof = n - 2
    
    # Mean of log_r
    log_r_mean = np.mean(log_r)
    
    # Sum of squared deviations
    ss_x = np.sum((log_r - log_r_mean)**2)
    
    # Mean squared error
    mse = np.sum(residuals**2) / dof
    
    # t-value
    t_val = stats.t.ppf(1 - (1 - confidence) / 2, dof)
    
    # Prediction band
    r_fine = np.linspace(log_r.min(), log_r.max(), 100)
    y_pred = slope * r_fine + intercept
    
    # Standard error of prediction
    se_pred = np.sqrt(mse * (1 + 1/n + (r_fine - log_r_mean)**2 / ss_x))
    
    # Confidence band (for mean response)
    se_conf = np.sqrt(mse * (1/n + (r_fine - log_r_mean)**2 / ss_x))
    
    y_lower = y_pred - t_val * se_conf
    y_upper = y_pred + t_val * se_conf
    
    return r_fine, y_pred, y_lower, y_upper


# =============================================================================
# Statistics Report
# =============================================================================

def compute_statistics(all_data, fit_results):
    """Compute comprehensive statistics and return formatted report."""
    lines = []
    lines.append("=" * 75)
    lines.append("MASS-RADIUS SCALING ANALYSIS REPORT")
    lines.append("=" * 75)
    lines.append("")
    lines.append("Fractal dimension extraction via mass-radius scaling:")
    lines.append("    M(R) ~ R^D  →  log M = D log R + const")
    lines.append("")
    lines.append("Theoretical value for 2D DLA: D = 1.71 ± 0.01")
    lines.append("")
    lines.append("Method: Ordinary least squares regression on log-transformed data")
    lines.append("        with bootstrap resampling for confidence intervals (95%).")
    lines.append("")
    
    # Summary table
    lines.append("=" * 75)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"  {'Case':<25} {'D':>8} {'±err':>8} {'R²':>8} {'|D-1.71|':>10}")
    lines.append("  " + "-" * 60)
    
    for case_idx, (filename, case_name) in enumerate(CASES):
        fit = fit_results[case_idx]
        if fit is not None:
            deviation = abs(fit['D'] - D_THEORY)
            lines.append(f"  {case_name:<25} {fit['D']:>8.4f} {fit['D_err']:>8.4f} "
                        f"{fit['R2']:>8.4f} {deviation:>10.4f}")
    lines.append("")
    
    # Detailed per-case analysis
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        fit = fit_results[case_idx]
        
        lines.append("=" * 75)
        lines.append(f"CASE {case_idx + 1}: {case_name.upper()}")
        lines.append(f"File: {filename}")
        lines.append("=" * 75)
        lines.append("")
        
        # Data properties
        lines.append("  DATA PROPERTIES:")
        lines.append(f"    Total particles:          {data['n_particles']:,}")
        lines.append(f"    Lattice size:             {data['lattice_size']} × {data['lattice_size']}")
        lines.append(f"    Stored D (from solver):   {data['fractal_dimension']:.4f}")
        lines.append("")
        
        if fit is not None:
            # Regression results
            lines.append("  LINEAR REGRESSION (log-log space):")
            lines.append(f"    Number of data points:    {fit['n_points']}")
            lines.append(f"    Radius range:             [{fit['r'].min():.2f}, {fit['r'].max():.2f}]")
            lines.append(f"    Mass range:               [{fit['m'].min():.0f}, {fit['m'].max():.0f}]")
            lines.append("")
            lines.append(f"    Slope (D):                {fit['D']:.6f}")
            lines.append(f"    Standard error:           {fit['D_std_err']:.6f}")
            lines.append(f"    95% CI:                   [{fit['D_ci_low']:.4f}, {fit['D_ci_high']:.4f}]")
            lines.append(f"    Bootstrap error:          ±{fit['D_err']:.4f}")
            lines.append("")
            lines.append(f"    Intercept (log c):        {fit['intercept']:.6f}")
            lines.append(f"    Prefactor c:              {np.exp(fit['intercept']):.6f}")
            lines.append("")
            
            # Goodness of fit
            lines.append("  GOODNESS OF FIT:")
            lines.append(f"    R² (coefficient):         {fit['R2']:.6f}")
            lines.append(f"    Adjusted R²:              {1 - (1 - fit['R2']) * (fit['n_points'] - 1) / (fit['n_points'] - 2):.6f}")
            lines.append(f"    p-value (slope ≠ 0):      {fit['p_value']:.2e}")
            lines.append(f"    RMSE (log space):         {fit['rmse']:.6f}")
            lines.append("")
            
            # Comparison with theory
            lines.append("  COMPARISON WITH THEORY (D = 1.71):")
            deviation = fit['D'] - D_THEORY
            z_score = deviation / fit['D_err']
            p_deviation = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            lines.append(f"    Deviation:                {deviation:+.4f}")
            lines.append(f"    Relative deviation:       {100 * deviation / D_THEORY:+.2f}%")
            lines.append(f"    z-score:                  {z_score:.2f}")
            lines.append(f"    p-value (D = 1.71):       {p_deviation:.4f}")
            
            if p_deviation > 0.05:
                lines.append("    → Consistent with theoretical value (p > 0.05)")
            else:
                lines.append("    → Statistically different from theory (p < 0.05)")
            lines.append("")
            
            # Residual analysis
            lines.append("  RESIDUAL ANALYSIS:")
            resid = fit['residuals']
            lines.append(f"    Mean residual:            {resid.mean():.6f}")
            lines.append(f"    Std residual:             {resid.std():.6f}")
            lines.append(f"    Max |residual|:           {np.abs(resid).max():.6f}")
            
            # Shapiro-Wilk normality test on residuals
            if len(resid) >= 3:
                stat_sw, p_sw = stats.shapiro(resid)
                sig = "***" if p_sw < 0.001 else "**" if p_sw < 0.01 else "*" if p_sw < 0.05 else ""
                lines.append(f"    Shapiro-Wilk W:           {stat_sw:.4f}")
                lines.append(f"    Shapiro-Wilk p:           {p_sw:.4f} {sig}")
                if p_sw > 0.05:
                    lines.append("    → Residuals consistent with normality")
                else:
                    lines.append("    → Residuals deviate from normality")
            lines.append("")
        else:
            lines.append("  ERROR: Insufficient data for regression analysis")
            lines.append("")
    
    # Statistical comparisons
    lines.append("=" * 75)
    lines.append("STATISTICAL COMPARISONS")
    lines.append("=" * 75)
    lines.append("")
    
    # Extract valid D values
    D_values = [f['D'] for f in fit_results if f is not None]
    D_errors = [f['D_err'] for f in fit_results if f is not None]
    
    if len(D_values) >= 2:
        lines.append("  ACROSS-CASE COMPARISON:")
        lines.append(f"    Mean D:                   {np.mean(D_values):.4f}")
        lines.append(f"    Std D:                    {np.std(D_values):.4f}")
        lines.append(f"    Range:                    [{min(D_values):.4f}, {max(D_values):.4f}]")
        lines.append(f"    Spread:                   {max(D_values) - min(D_values):.4f}")
        lines.append("")
        
        # Weighted mean
        weights = [1/e**2 for e in D_errors]
        weighted_mean = np.average(D_values, weights=weights)
        weighted_err = 1 / np.sqrt(sum(weights))
        lines.append(f"    Weighted mean D:          {weighted_mean:.4f} ± {weighted_err:.4f}")
        lines.append("")
        
        # Chi-square test for consistency
        chi2 = sum([(d - weighted_mean)**2 / e**2 for d, e in zip(D_values, D_errors)])
        dof = len(D_values) - 1
        p_chi2 = 1 - stats.chi2.cdf(chi2, dof)
        lines.append(f"    χ² (consistency):         {chi2:.4f}")
        lines.append(f"    Degrees of freedom:       {dof}")
        lines.append(f"    p-value:                  {p_chi2:.4f}")
        if p_chi2 > 0.05:
            lines.append("    → All cases mutually consistent (p > 0.05)")
        else:
            lines.append("    → Significant variation between cases (p < 0.05)")
        lines.append("")
        
        # Comparison with theory
        lines.append("  COMPARISON WITH THEORY (D = 1.71):")
        chi2_theory = sum([(d - D_THEORY)**2 / e**2 for d, e in zip(D_values, D_errors)])
        p_theory = 1 - stats.chi2.cdf(chi2_theory, len(D_values))
        lines.append(f"    χ² (vs theory):           {chi2_theory:.4f}")
        lines.append(f"    p-value:                  {p_theory:.4f}")
        if p_theory > 0.05:
            lines.append("    → All results consistent with D = 1.71 (p > 0.05)")
        else:
            lines.append("    → Some deviation from theoretical value (p < 0.05)")
    lines.append("")
    
    # Interpretation
    lines.append("=" * 75)
    lines.append("INTERPRETATION")
    lines.append("=" * 75)
    lines.append("")
    lines.append("  The fractal dimension D characterizes the scaling of mass with size")
    lines.append("  in the DLA aggregate. For 2D DLA, theory and extensive simulations")
    lines.append("  establish D ≈ 1.71, intermediate between a line (D=1) and a filled")
    lines.append("  disk (D=2), reflecting the ramified, dendritic morphology.")
    lines.append("")
    lines.append("  Key observations:")
    
    # Case-specific interpretation
    for case_idx, (filename, case_name) in enumerate(CASES):
        fit = fit_results[case_idx]
        if fit is not None:
            deviation = abs(fit['D'] - D_THEORY)
            if deviation < 0.02:
                assessment = "excellent"
            elif deviation < 0.05:
                assessment = "good"
            elif deviation < 0.10:
                assessment = "moderate"
            else:
                assessment = "poor"
            lines.append(f"    - {case_name}: D = {fit['D']:.3f} ({assessment} agreement)")
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


def plot_mass_radius_panel(ax, data, fit_result, panel_label):
    """Plot mass-radius data with fit for a single case."""
    if fit_result is None:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
               ha='center', va='center', fontsize=12)
        return
    
    log_r = fit_result['log_r']
    log_m = fit_result['log_m']
    D = fit_result['D']
    D_err = fit_result['D_err']
    R2 = fit_result['R2']
    
    # Plot data points
    ax.scatter(log_r, log_m, c=DATA_COLOR, s=40, alpha=0.7, 
              edgecolors='white', linewidths=0.5, zorder=3,
              label='Data')
    
    # Compute and plot confidence band
    r_fine, y_pred, y_lower, y_upper = compute_prediction_band(
        log_r, D, fit_result['intercept'], fit_result['residuals']
    )
    
    ax.fill_between(r_fine, y_lower, y_upper, color=BAND_COLOR, alpha=0.15,
                   label='95% CI', zorder=1)
    
    # Plot fit line
    ax.plot(r_fine, y_pred, color=FIT_COLOR, linewidth=2.5, 
           label=f'Fit: D = {D:.3f}±{D_err:.3f}', zorder=2)
    
    # Plot theoretical slope for reference
    y_theory = D_THEORY * r_fine + (fit_result['intercept'] + (D - D_THEORY) * np.mean(log_r))
    ax.plot(r_fine, y_theory, color=THEORY_COLOR, linewidth=1.5, 
           linestyle='--', alpha=0.7, label=f'Theory: D = {D_THEORY}', zorder=2)
    
    # Axis labels
    ax.set_xlabel(r'log R', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'log M(R)', fontsize=12, fontweight='bold')
    
    # Panel label
    ax.text(0.50, 1.05, panel_label,
           transform=ax.transAxes,
           fontsize=16,
           fontweight='bold',
           va='bottom',
           ha='center',
           color='black')
    
    # R² annotation
    ax.text(0.05, 0.95, f'R² = {R2:.4f}',
           transform=ax.transAxes,
           fontsize=10,
           fontweight='bold',
           va='top',
           ha='left',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # Bold tick labels
    for label in ax.xaxis.get_ticklabels():
        label.set_fontweight('bold')
    for label in ax.yaxis.get_ticklabels():
        label.set_fontweight('bold')
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def create_figure(all_data, fit_results):
    """Create the complete 2×2 panel figure."""
    setup_figure_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    axes = axes.flatten()
    
    for case_idx, (filename, case_name) in enumerate(CASES):
        ax = axes[case_idx]
        plot_mass_radius_panel(ax, all_data[case_idx], fit_results[case_idx], 
                              PANEL_LABELS[case_idx])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.5, w_pad=2.0)
    
    return fig


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 65)
    print("Mass-Radius Scaling Analysis")
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
        print(f"      → {len(data['radii'])} radius points, "
              f"stored D = {data['fractal_dimension']:.4f}")
    
    # Perform regression analysis
    print("\n[2/5] Fitting fractal dimensions...")
    fit_results = []
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        fit = fit_fractal_dimension(data['radii'], data['masses'])
        fit_results.append(fit)
        
        if fit is not None:
            print(f"      {case_name}: D = {fit['D']:.4f} ± {fit['D_err']:.4f} "
                  f"(R² = {fit['R2']:.4f})")
        else:
            print(f"      {case_name}: Insufficient data for fit")
    
    # Create figure
    print("\n[3/5] Creating figure...")
    fig = create_figure(all_data, fit_results)
    
    # Save figures
    print("\n[4/5] Saving figures...")
    
    # PNG
    png_path = FIGS_DIR / "mass_radius_scaling.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {png_path}")
    
    # PDF
    pdf_path = FIGS_DIR / "mass_radius_scaling.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {pdf_path}")
    
    # EPS
    eps_path = FIGS_DIR / "mass_radius_scaling.eps"
    fig.savefig(eps_path, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {eps_path}")
    
    plt.close(fig)
    
    # Generate and save report
    print("\n[5/5] Generating statistical report...")
    report = compute_statistics(all_data, fit_results)
    
    report_path = REPORTS_DIR / "mass_radius_scaling_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"      Saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"\nFigures saved to: {FIGS_DIR}")
    print(f"Report saved to: {REPORTS_DIR}")
    print(f"\nFractal Dimensions:")
    for case_idx, (_, case_name) in enumerate(CASES):
        fit = fit_results[case_idx]
        if fit:
            print(f"  {case_name}: D = {fit['D']:.4f} ± {fit['D_err']:.4f}")
    print(f"\nTheoretical value: D = {D_THEORY} ± {D_THEORY_ERR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
