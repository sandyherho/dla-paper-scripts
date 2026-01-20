#!/usr/bin/env python
"""
Publication-quality spatiotemporal snapshots of DLA aggregate growth.

Creates a 4×3 panel figure:
  - Rows: Case 1 (Classic), Case 2 (Multiple Seeds), Case 3 (Radial), Case 4 (High Density)
  - Columns: Early (10% growth), Middle (50% growth), End (100% growth)

Each panel displays the 2D lattice with aggregate particles colored by
distance from centroid, providing visual indication of growth history.

NOTE: We skip t=0 (seeds only) and start at early growth (~10%) to show
      meaningful aggregate structure in all panels.

Output:
  - ../figs/spatiotemporal_snapshots.png (300 dpi)
  - ../figs/spatiotemporal_snapshots.pdf (vector)
  - ../figs/spatiotemporal_snapshots.eps (vector)
  - ../reports/spatiotemporal_snapshots_report.txt

Author: Sandy H. S. Herho
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
from pathlib import Path
from scipy import ndimage
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
FIG_HEIGHT = 16
DPI = 300

# Panel labels
PANEL_LABELS = [
    "(a)", "(b)", "(c)",
    "(d)", "(e)", "(f)",
    "(g)", "(h)", "(i)",
    "(j)", "(k)", "(l)",
]

# Time points as fraction of total snapshots (skip beginning, start at 10%)
TIME_FRACTIONS = [0.10, 0.50, 1.0]  # Early (10%), Middle (50%), End (100%)
TIME_LABELS = ["Early", "Middle", "End"]

# Fire/inferno colormap matching the reference image
BACKGROUND_COLOR = '#000000'  # Pure black background


def create_fire_colormap():
    """
    Create custom fire colormap matching the reference DLA visualization.
    Dark center → deep red → orange → bright orange at edges.
    """
    colors = [
        (0.0, '#1a0000'),    # Very dark red (near center)
        (0.15, '#330000'),   # Dark red
        (0.3, '#660000'),    # Deep red
        (0.45, '#993300'),   # Red-brown
        (0.6, '#cc4400'),    # Dark orange
        (0.75, '#e65c00'),   # Orange
        (0.85, '#ff6600'),   # Bright orange
        (0.95, '#ff8533'),   # Light orange
        (1.0, '#ffaa66'),    # Pale orange (tips)
    ]
    
    positions = [c[0] for c in colors]
    hex_colors = [c[1] for c in colors]
    
    # Convert hex to RGB
    rgb_colors = []
    for h in hex_colors:
        h = h.lstrip('#')
        rgb_colors.append(tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4)))
    
    return LinearSegmentedColormap.from_list('fire_dla', 
                                              list(zip(positions, rgb_colors)), 
                                              N=256)


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
            'injection_mode': nc.injection_mode,
            'scenario': nc.scenario,
        }
        
        # Handle optional injection_radius
        if hasattr(nc, 'injection_radius'):
            data['injection_radius'] = nc.injection_radius
        else:
            data['injection_radius'] = None
    
    return data


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_aggregate_properties(grid, center_x, center_y):
    """Compute geometric properties of the aggregate."""
    N = grid.shape[0]
    
    # Find aggregate particles
    agg_mask = (grid == 2)
    n_particles = np.sum(agg_mask)
    
    if n_particles == 0:
        return {
            'n_particles': 0,
            'radius_max': 0,
            'radius_mean': 0,
            'radius_gyration': 0,
            'aspect_ratio': 1.0,
            'compactness': 0,
        }
    
    # Get coordinates of aggregate particles
    coords = np.argwhere(agg_mask)
    
    # Distance from center
    distances = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
    
    # Maximum radius (extent)
    radius_max = distances.max()
    
    # Mean radius
    radius_mean = distances.mean()
    
    # Radius of gyration
    radius_gyration = np.sqrt(np.mean(distances**2))
    
    # Bounding box and aspect ratio
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    aspect_ratio = max(width, height) / max(min(width, height), 1)
    
    # Compactness: ratio of particles to bounding circle area
    circle_area = np.pi * radius_max**2
    compactness = n_particles / max(circle_area, 1)
    
    return {
        'n_particles': n_particles,
        'radius_max': radius_max,
        'radius_mean': radius_mean,
        'radius_gyration': radius_gyration,
        'aspect_ratio': aspect_ratio,
        'compactness': compactness,
        'bounding_box': (width, height),
    }


def compute_distance_field(grid, center_x, center_y):
    """Compute distance from center for each aggregate particle."""
    N = grid.shape[0]
    distance_field = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        for j in range(N):
            if grid[i, j] == 2:
                distance_field[i, j] = np.sqrt((i - center_x)**2 + (j - center_y)**2)
    
    return distance_field


# =============================================================================
# Statistics Report
# =============================================================================

def compute_statistics(all_data, time_indices):
    """Compute comprehensive statistics and return formatted report."""
    lines = []
    lines.append("=" * 75)
    lines.append("SPATIOTEMPORAL SNAPSHOTS ANALYSIS REPORT")
    lines.append("=" * 75)
    lines.append("")
    lines.append("Generated for 4×3 panel figure (4 cases × 3 time points)")
    lines.append("Rows: Case 1 (Classic), Case 2 (Multiple Seeds),")
    lines.append("      Case 3 (Radial Injection), Case 4 (High Density)")
    lines.append("Columns: Early (~10%), Middle (~50%), End (100%) of growth")
    lines.append("")
    lines.append("NOTE: We skip t=0 (seeds only) to show meaningful aggregate")
    lines.append("      structure in all panels.")
    lines.append("")
    lines.append("All quantities in dimensionless lattice units unless specified.")
    lines.append("")
    
    # Global summary
    lines.append("=" * 75)
    lines.append("GLOBAL SUMMARY")
    lines.append("=" * 75)
    lines.append("")
    
    all_D = [d['fractal_dimension'] for d in all_data]
    all_N = [d['n_particles'] for d in all_data]
    
    lines.append(f"  Total cases analyzed:        4")
    lines.append(f"  Lattice size:                {all_data[0]['lattice_size']} × {all_data[0]['lattice_size']}")
    lines.append(f"  Fractal dimension range:     [{min(all_D):.4f}, {max(all_D):.4f}]")
    lines.append(f"  Particle count range:        [{min(all_N):,}, {max(all_N):,}]")
    lines.append(f"  Theoretical D (2D DLA):      1.71 ± 0.01")
    lines.append("")
    
    # Per-case detailed statistics
    for case_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[case_idx]
        
        lines.append("=" * 75)
        lines.append(f"CASE {case_idx + 1}: {case_name.upper()}")
        lines.append(f"File: {filename}")
        lines.append("=" * 75)
        lines.append("")
        
        # Simulation parameters
        lines.append("  SIMULATION PARAMETERS:")
        lines.append(f"    Lattice size N:           {data['lattice_size']}")
        lines.append(f"    Number of walkers:        {data['n_walkers']:,}")
        lines.append(f"    Number of seeds:          {data['n_seeds']}")
        lines.append(f"    Total iterations:         {data['n_iterations']:,}")
        lines.append(f"    Injection mode:           {data['injection_mode']}")
        if data['injection_radius'] is not None:
            lines.append(f"    Injection radius:         {data['injection_radius']:.1f}")
        lines.append("")
        
        # Final results
        lines.append("  FINAL AGGREGATE PROPERTIES:")
        lines.append(f"    Particles stuck:          {data['n_particles']:,}")
        lines.append(f"    Number of aggregates:     {data['n_aggregates']}")
        lines.append(f"    Fractal dimension D:      {data['fractal_dimension']:.4f}")
        lines.append(f"    Deviation from theory:    {abs(data['fractal_dimension'] - 1.71):.4f}")
        lines.append(f"    Center (x, y):            ({data['center_x']}, {data['center_y']})")
        lines.append("")
        
        # Temporal evolution
        lines.append("  TEMPORAL EVOLUTION:")
        lines.append("  " + "-" * 50)
        
        n_snapshots = len(data['snapshots'])
        t_indices = time_indices[case_idx]
        
        for col_idx, t_idx in enumerate(t_indices):
            panel_label = PANEL_LABELS[case_idx * 3 + col_idx]
            snapshot = data['snapshots'][t_idx]
            count = data['glued_counts'][t_idx]
            progress = 100 * count / data['n_walkers']
            
            props = compute_aggregate_properties(
                snapshot, data['center_x'], data['center_y']
            )
            
            lines.append(f"    {panel_label} {TIME_LABELS[col_idx]} (snapshot {t_idx + 1}/{n_snapshots}, {progress:.1f}%):")
            lines.append(f"        Particles:            {count:,}")
            lines.append(f"        Progress:             {progress:.1f}%")
            lines.append(f"        Maximum radius:       {props['radius_max']:.2f}")
            lines.append(f"        Mean radius:          {props['radius_mean']:.2f}")
            lines.append(f"        Radius of gyration:   {props['radius_gyration']:.2f}")
            lines.append(f"        Aspect ratio:         {props['aspect_ratio']:.3f}")
            lines.append(f"        Compactness:          {props['compactness']:.4f}")
            lines.append("")
        
        # Growth rate analysis
        counts = data['glued_counts']
        if len(counts) > 1:
            growth_rates = np.diff(counts)
            lines.append("  GROWTH RATE STATISTICS:")
            lines.append(f"    Mean particles/snapshot:  {growth_rates.mean():.2f}")
            lines.append(f"    Std particles/snapshot:   {growth_rates.std():.2f}")
            lines.append(f"    Max particles/snapshot:   {growth_rates.max()}")
            lines.append(f"    Min particles/snapshot:   {growth_rates.min()}")
            lines.append("")
    
    # Comparative analysis
    lines.append("=" * 75)
    lines.append("COMPARATIVE ANALYSIS")
    lines.append("=" * 75)
    lines.append("")
    
    # Rank by fractal dimension
    lines.append("  Ranking by Fractal Dimension (closest to 1.71):")
    sorted_by_D = sorted(enumerate(all_data), 
                         key=lambda x: abs(x[1]['fractal_dimension'] - 1.71))
    for rank, (idx, d) in enumerate(sorted_by_D, 1):
        lines.append(f"    {rank}. {CASES[idx][1]}: D = {d['fractal_dimension']:.4f} "
                    f"(Δ = {abs(d['fractal_dimension'] - 1.71):.4f})")
    lines.append("")
    
    # Rank by particle count
    lines.append("  Ranking by Final Particle Count:")
    sorted_by_N = sorted(enumerate(all_data), 
                         key=lambda x: x[1]['n_particles'], reverse=True)
    for rank, (idx, d) in enumerate(sorted_by_N, 1):
        lines.append(f"    {rank}. {CASES[idx][1]}: N = {d['n_particles']:,}")
    lines.append("")
    
    # Interpretation
    lines.append("=" * 75)
    lines.append("INTERPRETATION")
    lines.append("=" * 75)
    lines.append("")
    lines.append("  1. FRACTAL DIMENSION CONSISTENCY:")
    mean_D = np.mean(all_D)
    std_D = np.std(all_D)
    lines.append(f"     Mean D across cases: {mean_D:.4f} ± {std_D:.4f}")
    if abs(mean_D - 1.71) < 0.05:
        lines.append("     → Excellent agreement with theoretical value (1.71)")
    elif abs(mean_D - 1.71) < 0.10:
        lines.append("     → Good agreement with theoretical value (1.71)")
    else:
        lines.append("     → Moderate deviation from theoretical value (1.71)")
    lines.append("")
    
    lines.append("  2. GROWTH MORPHOLOGY:")
    lines.append("     - Classic DLA: Isotropic dendritic growth from single seed")
    lines.append("     - Multiple Seeds: Competitive growth with cluster coalescence")
    lines.append("     - Radial Injection: More uniform, isotropic growth pattern")
    lines.append("     - High Density: Denser structure from increased walker flux")
    lines.append("")
    
    lines.append("  3. SCREENING EFFECT:")
    lines.append("     The characteristic branched morphology arises from diffusive")
    lines.append("     screening: protruding tips intercept random walkers before")
    lines.append("     they can penetrate into fjords, amplifying initial perturbations.")
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
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': BACKGROUND_COLOR,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'text.usetex': False,
        'mathtext.fontset': 'dejavuserif',
    })


def plot_aggregate_snapshot(ax, grid, center_x, center_y):
    """Plot a single aggregate snapshot with distance-based fire coloring."""
    N = grid.shape[0]
    
    # Create RGB image with black background
    display = np.zeros((N, N, 3))
    
    # Get custom fire colormap
    cmap = create_fire_colormap()
    
    # Find aggregate particles and compute distances
    agg_mask = (grid == 2)
    if not np.any(agg_mask):
        ax.imshow(display, origin='lower', interpolation='nearest')
        return
    
    # Compute distance field
    distance_field = compute_distance_field(grid, center_x, center_y)
    max_dist = distance_field.max()
    
    if max_dist > 0:
        # Normalize distances
        norm_dist = distance_field / max_dist
        
        # Color aggregate particles
        for i in range(N):
            for j in range(N):
                if grid[i, j] == 2:
                    color = cmap(norm_dist[i, j])
                    display[i, j, :] = color[:3]
    
    # Display with interpolation for smoother appearance
    ax.imshow(display, origin='lower', interpolation='bilinear')
    
    # Set axis properties
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    
    # Axis labels
    ax.set_xlabel('x', fontsize=11, fontweight='bold')
    ax.set_ylabel('y', fontsize=11, fontweight='bold')
    
    # Bold tick labels
    for label in ax.xaxis.get_ticklabels():
        label.set_fontweight('bold')
    for label in ax.yaxis.get_ticklabels():
        label.set_fontweight('bold')
    
    # Reduce number of ticks for cleaner appearance
    ax.set_xticks([0, N//2, N])
    ax.set_yticks([0, N//2, N])


def create_figure(all_data):
    """Create the complete 4×3 panel figure."""
    setup_figure_style()
    
    fig, axes = plt.subplots(4, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    
    # Store time indices for report
    time_indices = []
    
    panel_idx = 0
    
    for row_idx, (filename, case_name) in enumerate(CASES):
        data = all_data[row_idx]
        n_snapshots = len(data['snapshots'])
        
        # Time indices based on fractions (skip t=0, start at early growth)
        # Early (10%), Middle (50%), End (100%)
        case_time_indices = []
        for frac in TIME_FRACTIONS:
            idx = min(int(frac * (n_snapshots - 1)), n_snapshots - 1)
            idx = max(idx, 1)  # Ensure we skip index 0 (just seeds)
            case_time_indices.append(idx)
        
        time_indices.append(case_time_indices)
        
        for col_idx, t_idx in enumerate(case_time_indices):
            ax = axes[row_idx, col_idx]
            
            # Get snapshot
            snapshot = data['snapshots'][t_idx]
            
            # Plot aggregate with fire colormap
            plot_aggregate_snapshot(
                ax, snapshot, 
                data['center_x'], data['center_y']
            )
            
            # Add panel label
            label = PANEL_LABELS[panel_idx]
            ax.text(
                0.50, 1.03, label,
                transform=ax.transAxes,
                fontsize=16,
                fontweight='bold',
                va='bottom',
                ha='center',
                color='black',
            )
            
            # Add particle count annotation
            count = data['glued_counts'][t_idx]
            progress = 100 * count / data['n_walkers']
            ax.text(
                0.98, 0.02, f'N = {count:,}\n({progress:.0f}%)',
                transform=ax.transAxes,
                fontsize=9,
                fontweight='bold',
                va='bottom',
                ha='right',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
            )
            
            panel_idx += 1
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=1.5, w_pad=1.0)
    
    return fig, time_indices


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 65)
    print("Spatiotemporal Snapshots Visualization")
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
        print(f"      → N={data['lattice_size']}, "
              f"particles={data['n_particles']:,}, "
              f"D={data['fractal_dimension']:.3f}")
    
    # Create figure
    print("\n[2/4] Creating figure...")
    fig, time_indices = create_figure(all_data)
    
    # Save figures
    print("\n[3/4] Saving figures...")
    
    # PNG
    png_path = FIGS_DIR / "spatiotemporal_snapshots.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {png_path}")
    
    # PDF
    pdf_path = FIGS_DIR / "spatiotemporal_snapshots.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {pdf_path}")
    
    # EPS
    eps_path = FIGS_DIR / "spatiotemporal_snapshots.eps"
    fig.savefig(eps_path, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"      Saved: {eps_path}")
    
    plt.close(fig)
    
    # Generate and save report
    print("\n[4/4] Generating statistical report...")
    report = compute_statistics(all_data, time_indices)
    
    report_path = REPORTS_DIR / "spatiotemporal_snapshots_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"      Saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"\nFigures saved to: {FIGS_DIR}")
    print(f"Report saved to: {REPORTS_DIR}")
    print(f"\nPanel layout (4×3):")
    print("  Rows: Classic DLA, Multiple Seeds, Radial Injection, High Density")
    print("  Cols: Early (~10%), Middle (~50%), End (100%)")
    print(f"\nColormap: Custom fire (dark red → orange, distance from center)")
    print("\nDone!")


if __name__ == "__main__":
    main()
