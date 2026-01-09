"""Cash flow decomposition and outlier analysis visualization."""

from __future__ import annotations
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Optional

from cashflow.engine import ForecastEngine, ForecastConfig
from cashflow.pipeline import clean_utf, detect_transfers, net_transfers, aggregate_monthly
from cashflow.pipeline.decomposition import decompose_cashflow
from cashflow.outliers.treatment import apply_residual_treatment

# Import from visualize_forecast
from visualize_forecast import generate_synthetic_data, COLORS, DPI

FIG_SIZE = (14, 10)
FIG_SIZE_WIDE = (16, 5)


def prepare_decomposed_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare decomposed and outlier-treated data."""
    utf_df = generate_synthetic_data(seed=seed)
    utf_df = clean_utf(utf_df)

    # Detect and net transfers
    utf_df = detect_transfers(utf_df, date_tolerance_days=2)
    external_df, transfer_summary = net_transfers(utf_df)

    # Aggregate monthly
    monthly_df = aggregate_monthly(external_df)

    # Decompose
    decomposed_df = decompose_cashflow(monthly_df, external_df)

    # Apply outlier treatment
    treated_df = apply_residual_treatment(
        decomposed_df,
        detection_method="modified_zscore",
        detection_threshold=3.5,
        treatment_method="median"
    )

    return treated_df, transfer_summary


def plot_decomposition_panels(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Three-panel plot showing NECF decomposition."""
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=True)

    months = pd.to_datetime(df["month_key"].astype(str) + "-01")

    # Panel 1: Total NECF
    ax1 = axes[0]
    ax1.plot(months, df["necf"], color=COLORS["actual"], linewidth=2.5,
             marker="o", markersize=5, label="Net External Cash Flow")
    ax1.fill_between(months, 0, df["necf"], alpha=0.3, color=COLORS["actual"])
    ax1.axhline(y=0, color="#6C757D", linewidth=1, linestyle="-")
    ax1.axhline(y=df["necf"].mean(), color=COLORS["actual"], linewidth=2,
                linestyle="--", alpha=0.7, label=f"Mean: {df['necf'].mean():.0f} EUR")

    ax1.set_ylabel("NECF (EUR)", fontsize=11, fontweight="bold")
    ax1.set_title("Cash Flow Decomposition Analysis", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#FAFAFA")

    # Panel 2: Deterministic Base
    ax2 = axes[1]
    ax2.plot(months, df["deterministic_base"], color=COLORS["deterministic"],
             linewidth=2.5, marker="s", markersize=5, label="Deterministic Base")
    ax2.fill_between(months, 0, df["deterministic_base"], alpha=0.3,
                     color=COLORS["deterministic"])
    ax2.axhline(y=0, color="#6C757D", linewidth=1, linestyle="-")
    ax2.axhline(y=df["deterministic_base"].mean(), color=COLORS["deterministic"],
                linewidth=2, linestyle="--", alpha=0.7,
                label=f"Mean: {df['deterministic_base'].mean():.0f} EUR")

    ax2.set_ylabel("Deterministic (EUR)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#FAFAFA")

    # Panel 3: Residual (with outliers highlighted)
    ax3 = axes[2]

    # Plot original residual
    residual_col = "residual_original" if "residual_original" in df.columns else "residual"
    ax3.plot(months, df[residual_col], color=COLORS["residual"], linewidth=2.5,
             marker="^", markersize=5, label="Residual (Original)", alpha=0.7)

    # Plot treated residual if available
    if "residual_clean" in df.columns:
        ax3.plot(months, df["residual_clean"], color=COLORS["forecast"], linewidth=2,
                 marker="v", markersize=5, label="Residual (Treated)", linestyle="--")

    ax3.axhline(y=0, color="#6C757D", linewidth=1, linestyle="-")

    # Highlight outliers
    if "is_outlier" in df.columns:
        outliers = df[df["is_outlier"]]
        if len(outliers) > 0:
            outlier_months = pd.to_datetime(outliers["month_key"].astype(str) + "-01")
            outlier_values = outliers[residual_col].values
            ax3.scatter(outlier_months, outlier_values, color=COLORS["outlier"],
                       s=150, zorder=5, marker="o", edgecolors="white",
                       linewidths=2, label="Detected Outliers")

    ax3.set_ylabel("Residual (EUR)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#FAFAFA")

    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_outlier_analysis(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Detailed outlier analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    months = pd.to_datetime(df["month_key"].astype(str) + "-01")
    residual_col = "residual_original" if "residual_original" in df.columns else "residual"

    # Panel 1: Residual time series with outlier zones
    ax1 = axes[0, 0]
    ax1.plot(months, df[residual_col], color=COLORS["actual"], linewidth=2,
             marker="o", markersize=5, label="Residual")

    # Calculate detection bounds (Modified Z-Score threshold)
    median = df[residual_col].median()
    mad = np.median(np.abs(df[residual_col] - median))
    threshold_upper = median + (3.5 * mad / 0.6745)
    threshold_lower = median - (3.5 * mad / 0.6745)

    ax1.axhline(y=threshold_upper, color=COLORS["outlier"], linewidth=2,
                linestyle="--", label=f"Upper Threshold")
    ax1.axhline(y=threshold_lower, color=COLORS["outlier"], linewidth=2,
                linestyle="--", label=f"Lower Threshold")
    ax1.axhline(y=median, color=COLORS["forecast"], linewidth=2,
                linestyle="-", alpha=0.7, label=f"Median: {median:.0f}")

    # Shade outlier zones
    ax1.fill_between(months, threshold_upper, df[residual_col].max() * 1.1,
                     alpha=0.1, color=COLORS["outlier"])
    ax1.fill_between(months, threshold_lower, df[residual_col].min() * 1.1,
                     alpha=0.1, color=COLORS["outlier"])

    if "is_outlier" in df.columns:
        outliers = df[df["is_outlier"]]
        if len(outliers) > 0:
            outlier_months = pd.to_datetime(outliers["month_key"].astype(str) + "-01")
            ax1.scatter(outlier_months, outliers[residual_col], color=COLORS["outlier"],
                       s=150, zorder=5, marker="o", edgecolors="white", linewidths=2)

    ax1.set_title("Outlier Detection (Modified Z-Score)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Month", fontsize=10)
    ax1.set_ylabel("Residual (EUR)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#FAFAFA")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Panel 2: Before vs After treatment
    ax2 = axes[0, 1]
    if "residual_clean" in df.columns and "is_outlier" in df.columns:
        outliers = df[df["is_outlier"]]
        if len(outliers) > 0:
            x = range(len(outliers))
            width = 0.35

            bars1 = ax2.bar([i - width/2 for i in x], outliers[residual_col],
                           width, label="Original", color=COLORS["outlier"],
                           edgecolor="white", linewidth=1.5)
            bars2 = ax2.bar([i + width/2 for i in x], outliers["residual_clean"],
                           width, label="Treated", color=COLORS["forecast"],
                           edgecolor="white", linewidth=1.5)

            ax2.set_xticks(x)
            ax2.set_xticklabels(outliers["month_key"], rotation=45, ha="right")

            # Add value labels
            for bar in bars1:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
            for bar in bars2:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No outliers detected", ha="center", va="center",
                    fontsize=14, transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "No outlier treatment data", ha="center", va="center",
                fontsize=14, transform=ax2.transAxes)

    ax2.axhline(y=0, color="#6C757D", linewidth=1, linestyle="-")
    ax2.set_title("Outlier Treatment: Before vs After", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Outlier Month", fontsize=10)
    ax2.set_ylabel("Residual Value (EUR)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#FAFAFA")

    # Panel 3: Residual distribution
    ax3 = axes[1, 0]
    ax3.hist(df[residual_col], bins=15, color=COLORS["actual"], edgecolor="white",
             linewidth=1.5, alpha=0.7, label="Original")
    if "residual_clean" in df.columns:
        ax3.hist(df["residual_clean"], bins=15, color=COLORS["forecast"],
                 edgecolor="white", linewidth=1.5, alpha=0.5, label="Treated")

    ax3.axvline(x=median, color=COLORS["deterministic"], linewidth=2.5,
                linestyle="-", label=f"Median: {median:.0f}")
    ax3.axvline(x=df[residual_col].mean(), color=COLORS["residual"], linewidth=2.5,
                linestyle="--", label=f"Mean: {df[residual_col].mean():.0f}")

    ax3.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Residual Value (EUR)", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_facecolor("#FAFAFA")

    # Panel 4: Outlier score distribution (if available)
    ax4 = axes[1, 1]
    if "outlier_score" in df.columns:
        scores = df["outlier_score"].dropna()
        colors_bar = [COLORS["outlier"] if abs(s) > 3.5 else COLORS["actual"]
                      for s in scores]

        ax4.bar(range(len(scores)), np.abs(scores), color=colors_bar,
                edgecolor="white", linewidth=0.5)
        ax4.axhline(y=3.5, color=COLORS["outlier"], linewidth=2.5,
                    linestyle="--", label="Threshold (|MZ| > 3.5)")

        ax4.set_title("Modified Z-Scores by Month", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Month Index", fontsize=10)
        ax4.set_ylabel("|Modified Z-Score|", fontsize=10)
        ax4.legend(loc="upper right", fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No outlier scores available", ha="center", va="center",
                fontsize=14, transform=ax4.transAxes)

    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_transfer_netting_summary(
    df: pd.DataFrame,
    transfer_summary: dict,
    output_path: str,
) -> None:
    """Visualize transfer netting impact."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Transfer netting summary
    ax1 = axes[0]
    labels = ["Transfers\nRemoved", "Volume\nNetted (÷100)"]
    values = [
        transfer_summary.get("num_transfers_removed", 0),
        transfer_summary.get("total_volume_removed", 0) / 100
    ]
    colors = [COLORS["deterministic"], COLORS["residual"]]

    bars = ax1.bar(labels, values, color=colors, edgecolor="white", linewidth=2)

    for bar, val in zip(bars, [transfer_summary.get("num_transfers_removed", 0),
                               transfer_summary.get("total_volume_removed", 0)]):
        label = f"{val:,.0f}" if val > 100 else f"{val}"
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax1.set_title("Transfer Netting Summary", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count / Volume (scaled)", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("#FAFAFA")

    # Panel 2: Monthly NECF impact
    ax2 = axes[1]
    months = pd.to_datetime(df["month_key"].astype(str) + "-01")
    ax2.plot(months, df["necf"], color=COLORS["actual"], linewidth=2.5,
             marker="o", markersize=5, label="External Cash Flow (Post-Netting)")
    ax2.axhline(y=df["necf"].mean(), color=COLORS["forecast"], linewidth=2,
                linestyle="--", label=f"Mean: {df['necf'].mean():.0f} EUR")

    ax2.set_title("Monthly NECF After Transfer Netting", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Month", fontsize=10)
    ax2.set_ylabel("NECF (EUR)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#FAFAFA")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all decomposition and outlier analysis plots."""
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    print("Preparing decomposed data...")
    treated_df, transfer_summary = prepare_decomposed_data(seed=42)

    print("\nGenerating plots...")

    # Plot 1: Three-panel decomposition
    plot_decomposition_panels(
        treated_df,
        str(output_dir / "decomposition_panels.png")
    )

    # Plot 2: Outlier analysis
    plot_outlier_analysis(
        treated_df,
        str(output_dir / "outlier_analysis.png")
    )

    # Plot 3: Transfer netting summary
    plot_transfer_netting_summary(
        treated_df, transfer_summary,
        str(output_dir / "transfer_netting.png")
    )

    print("\nDone! Decomposition plots saved to:", output_dir)


if __name__ == "__main__":
    main()
