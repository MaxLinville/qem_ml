import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

def plot_counts_histogram(
    counts: Dict[str, int],
    title: str = "Quantum Circuit Results",
    filename: Optional[str] = None,
    show: bool = True,
    color: str = "blue"
) -> plt.Figure:
    """
    Plot a histogram of quantum circuit measurement counts.

    Args:
        counts: Dictionary of measurement results mapping bitstrings to counts
        title: Title for the plot
        filename: If provided, save the figure to this file path
        show: Whether to display the figure
        color: Bar color
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort keys by binary value
    sorted_items = sorted(counts.items(), key=lambda x: int(x[0], 2))
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Plot bars
    ax.bar(range(len(labels)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70)
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Counts")
    ax.set_title(title)

    plt.tight_layout()

    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Show figure if requested
    if show:
        plt.show()
    
    return fig

def compare_distributions(
    clean_counts: Optional[Dict[str, int]] = None,
    noisy_counts: Optional[Dict[str, int]] = None,
    mitigated_counts: Optional[Dict[str, int]] = None,
    title: str = "Quantum Circuit Results Comparison",
    filename: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare clean, noisy, and/or mitigated distributions.

    Args:
        clean_counts: Ideal/clean circuit counts
        noisy_counts: Noisy circuit counts
        mitigated_counts: Error-mitigated circuit counts
        title: Title for the plot
        filename: If provided, save the figure to this file path
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect all available distributions
    all_counts = {}
    if clean_counts:
        all_counts["Clean"] = clean_counts
    if noisy_counts:
        all_counts["Noisy"] = noisy_counts
    if mitigated_counts:
        all_counts["Mitigated"] = mitigated_counts
    
    # Get all bitstrings and sort them
    all_bitstrings = set()
    for counts in all_counts.values():
        all_bitstrings.update(counts.keys())
    all_bitstrings = sorted(all_bitstrings, key=lambda x: int(x, 2))
    
    # Set up bar positions
    bar_width = 0.8 / len(all_counts)
    colors = {"Clean": "blue", "Noisy": "red", "Mitigated": "green"}
    
    # Plot each distribution
    for i, (label, counts) in enumerate(all_counts.items()):
        # Normalize to probabilities
        total = sum(counts.values())
        values = [counts.get(bs, 0) / total for bs in all_bitstrings]
        
        # Calculate bar positions
        positions = np.arange(len(all_bitstrings)) + i * bar_width - (bar_width * len(all_counts) / 2) + bar_width/2
        
        # Plot bars
        ax.bar(positions, values, bar_width, label=label, color=colors.get(label, "gray"), alpha=0.7)
    
    # Configure plot
    ax.set_xticks(np.arange(len(all_bitstrings)))
    ax.set_xticklabels(all_bitstrings, rotation=70)
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Show figure if requested
    if show:
        plt.show()
    
    return fig

def plot_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Error Mitigation Metrics",
    filename: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot error metrics over iterations.

    Args:
        metrics: Dictionary mapping metric names to lists of values
        title: Title for the plot
        filename: If provided, save the figure to this file path
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    for name, values in metrics.items():
        ax.plot(values, 'o-', label=name)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    # Show figure if requested
    if show:
        plt.show()
        
    return fig