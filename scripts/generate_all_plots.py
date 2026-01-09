#!/usr/bin/env python
"""Generate all visualization plots for client presentation."""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")


def main():
    """Run all visualization scripts."""
    print("=" * 60)
    print("CASH FLOW FORECASTING - VISUALIZATION SUITE")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    # Import and run each visualization module
    print("\n" + "-" * 60)
    print("1. FORECAST VISUALIZATIONS")
    print("-" * 60)
    from visualize_forecast import main as forecast_main
    forecast_main()

    print("\n" + "-" * 60)
    print("2. SCENARIO COMPARISON")
    print("-" * 60)
    from visualize_scenarios import main as scenarios_main
    scenarios_main()

    print("\n" + "-" * 60)
    print("3. DECOMPOSITION & OUTLIER ANALYSIS")
    print("-" * 60)
    from visualize_decomposition import main as decomposition_main
    decomposition_main()

    # Summary
    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60)

    # List generated files
    plot_files = list(output_dir.glob("*.png"))
    print(f"\nGenerated {len(plot_files)} plots in: {output_dir}/")
    print()
    for f in sorted(plot_files):
        print(f"  - {f.name}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
