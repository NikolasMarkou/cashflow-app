"""Robustness analysis - comprehensive Phase 2 validation.

This script runs all Phase 2 validation scenarios:
- 2.1 Fat-tailed distributions (Student-t, Laplace, Mixture)
- 2.2 Regime shift scenarios
- 2.3 Transfer tolerance sweep

Outputs plots and summary to plots/robustness_analysis/
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

from analyze_noise_sensitivity import (
    NoiseConfig, NoiseResult,
    NOISE_LEVELS_GAUSSIAN,
    NOISE_LEVELS_STUDENT_T,
    NOISE_LEVELS_LAPLACE,
    NOISE_LEVELS_MIXTURE,
    NOISE_LEVELS_REGIME_SHIFT,
    TransferConfig, TRANSFER_CONFIGS,
    analyze_noise_level,
    run_transfer_tolerance_sweep,
)

# Plot settings
COLORS = {
    "gaussian": "#2E86AB",
    "student_t": "#A23B72",
    "laplace": "#F18F01",
    "mixture": "#C73E1D",
    "regime": "#28A745",
    "threshold": "#DC3545",
}
FIG_SIZE = (14, 8)
DPI = 150


def run_distribution_comparison(
    seeds: List[int] = None,
    num_seeds: int = 10
) -> Dict[str, List[NoiseResult]]:
    """Run analysis comparing different noise distributions.

    Tests WMAPE stability across Gaussian, Student-t, Laplace, and Mixture.
    """
    if seeds is None:
        seeds = list(range(num_seeds))

    results = {}

    # Reference Gaussian (moderate noise)
    gaussian_config = NOISE_LEVELS_GAUSSIAN[3]  # Moderate Noise
    print(f"  Analyzing: {gaussian_config.name} (Gaussian baseline)...")
    results["Gaussian"] = []
    for seed in seeds:
        try:
            result = analyze_noise_level(gaussian_config, seed=seed)
            results["Gaussian"].append(result)
        except Exception as e:
            print(f"    Warning: Seed {seed} failed - {e}")

    # Student-t distributions
    for config in NOISE_LEVELS_STUDENT_T:
        print(f"  Analyzing: {config.name}...")
        results[config.name] = []
        for seed in seeds:
            try:
                result = analyze_noise_level(config, seed=seed)
                results[config.name].append(result)
            except Exception as e:
                print(f"    Warning: Seed {seed} failed - {e}")

    # Laplace
    laplace_config = NOISE_LEVELS_LAPLACE[1]  # Laplace Moderate
    print(f"  Analyzing: {laplace_config.name}...")
    results["Laplace"] = []
    for seed in seeds:
        try:
            result = analyze_noise_level(laplace_config, seed=seed)
            results["Laplace"].append(result)
        except Exception as e:
            print(f"    Warning: Seed {seed} failed - {e}")

    # Mixture
    mixture_config = NOISE_LEVELS_MIXTURE[0]  # Mixture (5% extreme)
    print(f"  Analyzing: {mixture_config.name}...")
    results["Mixture"] = []
    for seed in seeds:
        try:
            result = analyze_noise_level(mixture_config, seed=seed)
            results["Mixture"].append(result)
        except Exception as e:
            print(f"    Warning: Seed {seed} failed - {e}")

    return results


def run_regime_shift_analysis(
    seeds: List[int] = None,
    num_seeds: int = 10
) -> Dict[str, List[NoiseResult]]:
    """Run analysis across regime shift scenarios."""
    if seeds is None:
        seeds = list(range(num_seeds))

    results = {}

    for config in NOISE_LEVELS_REGIME_SHIFT:
        print(f"  Analyzing: {config.name}...")
        results[config.name] = []
        for seed in seeds:
            try:
                result = analyze_noise_level(config, seed=seed)
                results[config.name].append(result)
            except Exception as e:
                print(f"    Warning: Seed {seed} failed - {e}")

    return results


def plot_distribution_comparison(
    results: Dict[str, List[NoiseResult]],
    output_path: str
) -> None:
    """Plot WMAPE comparison across distribution types."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    distributions = list(results.keys())
    means = []
    stds = []
    colors = []

    color_map = {
        "Gaussian": COLORS["gaussian"],
        "Student-t (df=3)": COLORS["student_t"],
        "Student-t (df=5)": COLORS["student_t"],
        "Student-t (df=10)": COLORS["student_t"],
        "Laplace": COLORS["laplace"],
        "Mixture": COLORS["mixture"],
    }

    for dist in distributions:
        wmapes = [r.wmape for r in results[dist]]
        means.append(np.mean(wmapes))
        stds.append(np.std(wmapes))
        colors.append(color_map.get(dist, "#6C757D"))

    x = np.arange(len(distributions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", linewidth=2, alpha=0.8)

    # Threshold line
    ax.axhline(y=20, color=COLORS["threshold"], linewidth=2.5,
               linestyle="--", label="Acceptance Threshold (20%)")

    # Value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.3,
               f"{mean:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Noise Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("WMAPE (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Accuracy by Noise Distribution Type\n(Phase 2.1: Fat-Tailed Distributions)",
                fontsize=14, fontweight="bold", pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(distributions, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, max(means) * 1.5)

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_regime_shift_accuracy(
    results: Dict[str, List[NoiseResult]],
    output_path: str
) -> None:
    """Plot WMAPE and pass rate for regime shift scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    scenarios = list(results.keys())
    means = []
    stds = []
    pass_rates = []

    for scenario in scenarios:
        wmapes = [r.wmape for r in results[scenario]]
        means.append(np.mean(wmapes))
        stds.append(np.std(wmapes))
        passes = sum(1 for r in results[scenario] if r.meets_threshold)
        pass_rates.append(100 * passes / len(results[scenario]) if results[scenario] else 0)

    x = np.arange(len(scenarios))

    # Left: WMAPE
    bars1 = ax1.bar(x, means, yerr=stds, capsize=4, color=COLORS["regime"],
                   edgecolor="white", linewidth=1.5, alpha=0.8)
    ax1.axhline(y=20, color=COLORS["threshold"], linewidth=2, linestyle="--")

    ax1.set_xlabel("Regime Shift Scenario", fontsize=11, fontweight="bold")
    ax1.set_ylabel("WMAPE (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Forecast Accuracy Under Regime Shifts", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("Regime: ", "") for s in scenarios],
                        fontsize=8, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("#FAFAFA")

    # Right: Pass rate
    colors = [COLORS["regime"] if pr >= 80 else ("#FFC107" if pr >= 50 else COLORS["threshold"])
              for pr in pass_rates]
    bars2 = ax2.bar(x, pass_rates, color=colors, edgecolor="white", linewidth=1.5, alpha=0.8)

    for bar, rate in zip(bars2, pass_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.axhline(y=80, color=COLORS["regime"], linewidth=1.5, linestyle="--", alpha=0.5)
    ax2.axhline(y=50, color="#FFC107", linewidth=1.5, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Regime Shift Scenario", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Pass Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Pass Rate (WMAPE < 20%) Under Regime Shifts", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("Regime: ", "") for s in scenarios],
                        fontsize=8, rotation=45, ha="right")
    ax2.set_ylim(0, 115)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#FAFAFA")

    fig.suptitle("Phase 2.2: Regime Shift Validation", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_transfer_tolerance_sweep(
    output_path: str,
    seeds: List[int] = None
) -> pd.DataFrame:
    """Run and plot transfer tolerance sweep analysis."""
    if seeds is None:
        seeds = list(range(10))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    all_results = []
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(TRANSFER_CONFIGS)))

    # Run sweep for each transfer config
    for cfg, color in zip(TRANSFER_CONFIGS, colors):
        print(f"  Analyzing: {cfg.name}...")
        df = run_transfer_tolerance_sweep(cfg, seeds=seeds)
        all_results.append(df)

        # Aggregate by tolerance
        agg = df.groupby("tolerance_days").agg({
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
        }).reset_index()

        tolerances = agg["tolerance_days"].values
        precision_mean = agg["precision"]["mean"].values
        recall_mean = agg["recall"]["mean"].values
        f1_mean = agg["f1"]["mean"].values

        # Plot precision
        axes[0, 0].plot(tolerances, precision_mean * 100, marker="o",
                       label=cfg.name, color=color, linewidth=2)
        # Plot recall
        axes[0, 1].plot(tolerances, recall_mean * 100, marker="s",
                       label=cfg.name, color=color, linewidth=2)
        # Plot F1
        axes[1, 0].plot(tolerances, f1_mean * 100, marker="^",
                       label=cfg.name, color=color, linewidth=2)

    # Style axes
    for ax, title, ylabel in [
        (axes[0, 0], "Precision by Tolerance", "Precision (%)"),
        (axes[0, 1], "Recall by Tolerance", "Recall (%)"),
        (axes[1, 0], "F1 Score by Tolerance", "F1 Score (%)"),
    ]:
        ax.set_xlabel("Tolerance (days)", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        ax.set_ylim(0, 105)

    # Summary table in bottom right
    axes[1, 1].axis("off")

    summary_text = "Optimal Tolerance by Scenario:\n\n"
    combined = pd.concat(all_results)
    for cfg in TRANSFER_CONFIGS:
        cfg_data = combined[combined["config_name"] == cfg.name]
        f1_by_tol = cfg_data.groupby("tolerance_days")["f1"].mean()
        optimal_tol = f1_by_tol.idxmax()
        optimal_f1 = f1_by_tol.max()
        summary_text += f"{cfg.name}:\n  Optimal: {optimal_tol}d (F1={optimal_f1:.1%})\n"

    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment="top", fontfamily="monospace",
                   bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))

    fig.suptitle("Phase 2.3: Transfer Tolerance Sweep Analysis", fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")

    return pd.concat(all_results)


def generate_summary_report(
    dist_results: Dict[str, List[NoiseResult]],
    regime_results: Dict[str, List[NoiseResult]],
    transfer_results: pd.DataFrame,
    output_path: str
) -> None:
    """Generate comprehensive summary CSV."""
    rows = []

    # Distribution comparison
    gaussian_mean = np.mean([r.wmape for r in dist_results.get("Gaussian", [])])
    for dist, results in dist_results.items():
        if not results:
            continue
        wmapes = [r.wmape for r in results]
        passes = sum(1 for r in results if r.meets_threshold)
        rows.append({
            "Phase": "2.1 Distributions",
            "Scenario": dist,
            "WMAPE Mean": np.mean(wmapes),
            "WMAPE Std": np.std(wmapes),
            "Pass Rate": 100 * passes / len(results),
            "vs Gaussian": np.mean(wmapes) - gaussian_mean if dist != "Gaussian" else 0,
        })

    # Regime shifts
    baseline_mean = np.mean([r.wmape for r in regime_results.get("Regime: Baseline", [])])
    for scenario, results in regime_results.items():
        if not results:
            continue
        wmapes = [r.wmape for r in results]
        passes = sum(1 for r in results if r.meets_threshold)
        rows.append({
            "Phase": "2.2 Regime Shifts",
            "Scenario": scenario.replace("Regime: ", ""),
            "WMAPE Mean": np.mean(wmapes),
            "WMAPE Std": np.std(wmapes),
            "Pass Rate": 100 * passes / len(results),
            "vs Baseline": np.mean(wmapes) - baseline_mean if "Baseline" not in scenario else 0,
        })

    # Transfer tolerance
    for cfg_name in transfer_results["config_name"].unique():
        cfg_data = transfer_results[transfer_results["config_name"] == cfg_name]
        f1_by_tol = cfg_data.groupby("tolerance_days")["f1"].mean()
        optimal_tol = f1_by_tol.idxmax()
        optimal_f1 = f1_by_tol.max()
        recall_at_2 = cfg_data[cfg_data["tolerance_days"] == 2]["recall"].mean()
        rows.append({
            "Phase": "2.3 Transfer Tolerance",
            "Scenario": cfg_name,
            "Optimal Tolerance": optimal_tol,
            "Optimal F1": optimal_f1 * 100,
            "Recall@2d": recall_at_2 * 100,
            "vs Baseline": 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    """Run complete robustness analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Robustness analysis for cashflow forecasting")
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        default=10,
        help="Number of random seeds to use (default: 10)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots"
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "plots" / "robustness_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))

    print("=" * 70)
    print("ROBUSTNESS ANALYSIS - Phase 2 Validation")
    print("=" * 70)

    # Phase 2.1: Distribution comparison
    print("\n--- Phase 2.1: Fat-Tailed Distribution Comparison ---")
    dist_results = run_distribution_comparison(seeds=seeds)
    plot_distribution_comparison(dist_results, str(output_dir / "distribution_comparison.png"))

    # Phase 2.2: Regime shift analysis
    print("\n--- Phase 2.2: Regime Shift Analysis ---")
    regime_results = run_regime_shift_analysis(seeds=seeds)
    plot_regime_shift_accuracy(regime_results, str(output_dir / "regime_shift_accuracy.png"))

    # Phase 2.3: Transfer tolerance sweep
    print("\n--- Phase 2.3: Transfer Tolerance Sweep ---")
    transfer_results = plot_transfer_tolerance_sweep(
        str(output_dir / "transfer_tolerance_sweep.png"),
        seeds=seeds
    )

    # Generate summary
    print("\n--- Generating Summary Report ---")
    generate_summary_report(
        dist_results, regime_results, transfer_results,
        str(output_dir / "robustness_summary.csv")
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")

    # Quick summary
    print("\n--- Quick Summary ---")

    print("\nPhase 2.1 (Distributions):")
    gaussian_wmape = np.mean([r.wmape for r in dist_results.get("Gaussian", [])])
    for dist, results in dist_results.items():
        if results:
            wmape = np.mean([r.wmape for r in results])
            diff = wmape - gaussian_wmape
            print(f"  {dist}: WMAPE = {wmape:.2f}% ({diff:+.2f}% vs Gaussian)")

    print("\nPhase 2.2 (Regime Shifts):")
    for scenario, results in regime_results.items():
        if results:
            passes = sum(1 for r in results if r.meets_threshold)
            pass_rate = 100 * passes / len(results)
            status = "PASS" if pass_rate >= 80 else "WARN" if pass_rate >= 50 else "FAIL"
            print(f"  {scenario.replace('Regime: ', '')}: Pass Rate = {pass_rate:.0f}% [{status}]")

    print("\nPhase 2.3 (Transfer Tolerance):")
    for cfg_name in transfer_results["config_name"].unique():
        cfg_data = transfer_results[transfer_results["config_name"] == cfg_name]
        f1_by_tol = cfg_data.groupby("tolerance_days")["f1"].mean()
        optimal_tol = f1_by_tol.idxmax()
        optimal_f1 = f1_by_tol.max()
        print(f"  {cfg_name}: Optimal tolerance = {optimal_tol}d (F1={optimal_f1:.1%})")


if __name__ == "__main__":
    main()
