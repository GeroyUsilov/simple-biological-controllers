"""
Visualization and Analysis Tools

This module provides functions for plotting and analyzing the results
of the robustness optimization simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import config


def setup_plotting_style():
    """
    Set up consistent plotting style for all figures.
    """
    plt.rcParams['font.family'] = config.FONT_FAMILY
    plt.rcParams['font.sans-serif'] = [config.FONT_NAME]
    plt.rcParams.update({'font.size': config.BASE_FONT_SIZE})
    plt.rcParams['axes.titlesize'] = config.BASE_FONT_SIZE
    plt.rcParams['axes.labelsize'] = config.BASE_FONT_SIZE
    plt.rcParams['xtick.labelsize'] = config.BASE_FONT_SIZE
    plt.rcParams['ytick.labelsize'] = config.BASE_FONT_SIZE
    plt.rcParams['legend.fontsize'] = config.BASE_FONT_SIZE


def plot_mode_gap_evolution(mode_gaps: List[float], costs: List[float], 
                           save_path: str = None, show_plot: bool = True):
    """
    Plot the evolution of mode gap during optimization.
    
    Args:
        mode_gaps: List of mode gap values over iterations
        costs: List of cost values over iterations
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    setup_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(config.FIGURE_WIDTH_IN, config.FIGURE_HEIGHT_IN))
    
    # Plot mode gap evolution
    ax1.plot(mode_gaps, label='Mode Gap', linewidth=2.0, color='tab:blue')
    ax1.set_ylabel('Mode Gap', fontsize=12)
    ax1.set_xlabel('Optimization Iteration', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cost evolution
    ax2.plot(costs, label='Robustness Cost', linewidth=2.0, color='tab:orange')
    ax2.set_ylabel('Robustness Cost', fontsize=12)
    ax2.set_xlabel('Optimization Iteration', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_robustness_distribution(controlled_distances: List[float], 
                                uncontrolled_distances: List[float],
                                save_path: str = None, show_plot: bool = True):
    """
    Plot the distribution of robustness measures for controlled vs uncontrolled systems.
    
    Args:
        controlled_distances: List of distances for controlled system
        uncontrolled_distances: List of distances for uncontrolled system
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    setup_plotting_style()
    
    # Filter out extreme values for better visualization
    controlled_array = np.array(controlled_distances)
    uncontrolled_array = np.array(uncontrolled_distances)
    
    mask = (controlled_array < 1) & (uncontrolled_array < 1)
    
    fig, ax = plt.subplots(figsize=(config.FIGURE_WIDTH_IN, config.FIGURE_HEIGHT_IN))
    
    # Plot histograms
    ax.hist(np.log10(uncontrolled_array[mask]), alpha=0.7, bins=50, 
            density=True, label='Uncontrolled', color='tab:red')
    ax.hist(np.log10(controlled_array[mask]), alpha=0.7, bins=50, 
            density=True, label='With Controller', color='tab:blue')
    
    ax.set_xlabel('Log10 Distance from Fixed Point', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set reasonable limits
    ax.set_xlim([-6, 0])
    ax.set_ylim([0, 0.9])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_robustness_tradeoff(mode_gaps: List[float], costs: List[float],
                            save_path: str = None, show_plot: bool = True):
    """
    Plot the trade-off between mode gap and robustness cost.
    
    Args:
        mode_gaps: List of mode gap values
        costs: List of cost values
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(config.FIGURE_WIDTH_IN, config.FIGURE_HEIGHT_IN))
    
    # Create scatter plot with different colors for different phases
    mode_gaps_array = np.array(mode_gaps)
    costs_array = np.array(costs)
    
    # Plot different phases with different colors
    n_points = len(mode_gaps)
    early_idx = slice(0, min(100, n_points), 15)
    middle_idx = slice(100, min(4500, n_points), 100)
    late_idx = slice(4500, n_points, 15)
    
    ax.scatter(mode_gaps_array[early_idx], costs_array[early_idx], 
              alpha=0.5, color="tab:blue", label='Early Optimization')
    ax.scatter(mode_gaps_array[middle_idx], costs_array[middle_idx], 
              alpha=0.5, color="tab:blue")
    ax.scatter(mode_gaps_array[late_idx], costs_array[late_idx], 
              alpha=0.5, color="tab:blue")
    ax.scatter(mode_gaps_array[-1], costs_array[-1], 
              alpha=0.8, color="tab:blue", s=100, marker='*')
    
    ax.set_xscale('log')
    ax.set_xlim((1, 100))
    ax.set_xlabel('Mode Gap', fontsize=12)
    ax.set_ylabel('Normalized Distance from Fixed Point', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_system_dynamics(n_eval: np.ndarray, t_eval: np.ndarray, n_f: np.ndarray,
                        title: str = "System Dynamics", save_path: str = None, 
                        show_plot: bool = True):
    """
    Plot the evolution of system states over time.
    
    Args:
        n_eval: State evolution matrix (dimensions x time)
        t_eval: Time points
        n_f: Fixed point
        title: Plot title
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(config.FIGURE_WIDTH_IN, config.FIGURE_HEIGHT_IN))
    
    # Plot deviation from fixed point for each component
    for i in range(n_eval.shape[0]):
        deviation = n_eval[i, :] - n_f[i]
        ax.plot(t_eval, deviation, alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Deviation from Fixed Point', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_eigenmode_analysis(w: np.ndarray, v: np.ndarray, n_f: np.ndarray,
                           save_path: str = None, show_plot: bool = True):
    """
    Plot eigenmode analysis of the system.
    
    Args:
        w: Eigenvalues
        v: Eigenvectors
        n_f: Fixed point
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    setup_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.FIGURE_WIDTH_IN * 2, config.FIGURE_HEIGHT_IN))
    
    # Plot eigenvalue spectrum
    sorted_indices = np.argsort(w)
    ax1.plot(range(len(w)), w[sorted_indices], 'o-', markersize=4)
    ax1.set_xlabel('Eigenvalue Index (sorted)', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Eigenvalue Spectrum', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot slowest mode projection
    slowest_mode = v[:, sorted_indices[-1]]
    projection = np.dot(slowest_mode, n_f)
    ax2.bar(range(len(slowest_mode)), np.abs(slowest_mode), alpha=0.7)
    ax2.set_xlabel('Component Index', fontsize=12)
    ax2.set_ylabel('|Slowest Mode Component|', fontsize=12)
    ax2.set_title(f'Slowest Mode (λ={w[sorted_indices[-1]]:.3f})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_optimization_summary(costs: List[float], mode_gaps: List[float],
                            controlled_distances: List[float] = None,
                            uncontrolled_distances: List[float] = None,
                            save_path: str = None, show_plot: bool = True):
    """
    Create a comprehensive summary plot of the optimization results.
    
    Args:
        costs: List of cost values
        mode_gaps: List of mode gap values
        controlled_distances: List of controlled distances (optional)
        uncontrolled_distances: List of uncontrolled distances (optional)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    setup_plotting_style()
    
    n_plots = 2
    if controlled_distances is not None and uncontrolled_distances is not None:
        n_plots = 3
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(config.FIGURE_WIDTH_IN, config.FIGURE_HEIGHT_IN * n_plots))
    
    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    
    # Plot 1: Optimization progress
    ax1.plot(costs, label='Robustness Cost', color='tab:orange', linewidth=2)
    ax1.set_ylabel('Cost', fontsize=12)
    ax1.set_title('Optimization Progress', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mode gap evolution
    ax2.plot(mode_gaps, label='Mode Gap', color='tab:blue', linewidth=2)
    ax2.set_ylabel('Mode Gap', fontsize=12)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_title('Mode Gap Evolution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Robustness distribution (if data available)
    if n_plots == 3:
        controlled_array = np.array(controlled_distances)
        uncontrolled_array = np.array(uncontrolled_distances)
        mask = (controlled_array < 1) & (uncontrolled_array < 1)
        
        ax3.hist(np.log10(uncontrolled_array[mask]), alpha=0.7, bins=30, 
                density=True, label='Uncontrolled', color='tab:red')
        ax3.hist(np.log10(controlled_array[mask]), alpha=0.7, bins=30, 
                density=True, label='Controlled', color='tab:blue')
        ax3.set_xlabel('Log10 Distance', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Robustness Distribution', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def print_optimization_summary(costs: List[float], mode_gaps: List[float],
                             final_k: np.ndarray, final_K: np.ndarray,
                             final_n_f: np.ndarray):
    """
    Print a summary of the optimization results.
    
    Args:
        costs: List of cost values
        mode_gaps: List of mode gap values
        final_k: Final k matrix
        final_K: Final K matrix
        final_n_f: Final fixed point
    """
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    
    print(f"Initial Cost: {costs[0]:.6f}")
    print(f"Final Cost: {costs[-1]:.6f}")
    print(f"Cost Improvement: {((costs[0] - costs[-1]) / costs[0] * 100):.2f}%")
    
    print(f"\nInitial Mode Gap: {mode_gaps[0]:.6f}")
    print(f"Final Mode Gap: {mode_gaps[-1]:.6f}")
    print(f"Mode Gap Change: {((mode_gaps[-1] - mode_gaps[0]) / mode_gaps[0] * 100):.2f}%")
    
    print(f"\nFinal Fixed Point Properties:")
    print(f"  Total Population: {np.sum(final_n_f):.4f}")
    print(f"  Population Range: [{np.min(final_n_f):.4f}, {np.max(final_n_f):.4f}]")
    print(f"  Population Std: {np.std(final_n_f):.4f}")
    
    print(f"\nNetwork Properties:")
    print(f"  k Matrix Sparsity: {(final_k == 0).sum() / final_k.size * 100:.1f}%")
    print(f"  K Matrix Sparsity: {(final_K == 0).sum() / final_K.size * 100:.1f}%")
    print(f"  k Matrix Mean: {np.mean(final_k):.4f}")
    print(f"  K Matrix Mean: {np.mean(final_K):.4f}")
    
    print("="*60) 