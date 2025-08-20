#!/usr/bin/env python3
"""
Example script demonstrating the Duality Robustness Simulation

This script shows how to use the refactored code to run a simple
robustness optimization simulation.
"""

import numpy as np
import config
from dynamical_system import DynamicalSystem
from optimizer import RobustnessOptimizer
from data_loader import create_simple_control_matrices
from visualization import plot_optimization_summary, print_optimization_summary


def run_simple_example():
    """
    Run a simple example with a random network and basic control.
    """
    print("="*60)
    print("SIMPLE EXAMPLE: Duality Robustness Simulation")
    print("="*60)
    
    # Create a simple random network
    d = config.SYSTEM_DIMENSION
    k = np.random.rand(d, d) * 0.1
    K = np.random.rand(d, d) * 0.1
    
    # Make it sparse
    mask = np.random.rand(d, d) > 0.8
    k[~mask] = 0
    K[~mask] = 0
    
    print(f"Created random network with {d}x{d} dimensions")
    print(f"Network sparsity: {(k == 0).sum() / k.size * 100:.1f}%")
    
    # Create dynamical system
    system = DynamicalSystem(
        k=k, K=K, lam=config.DECAY_VECTOR,
        k_p=config.PROPORTIONAL_GAIN, k_i=config.INTEGRAL_GAIN
    )
    
    # Create simple control matrices
    g, s = create_simple_control_matrices(d, config.NUM_CONTROL_CHANNELS)
    system.update_control_matrices(g, s)
    
    print(f"Created control system with {config.NUM_CONTROL_CHANNELS} control channels")
    
    # Find initial fixed point
    initial_state = np.zeros(d) + 0.5
    fixed_point = system.find_fixed_point(initial_state, config.SIMULATION_TIME)
    
    print(f"Found fixed point with total population: {np.sum(fixed_point):.4f}")
    print(f"Initial mode gap: {system.calc_mode_gap(fixed_point):.4f}")
    
    # Create optimizer
    optimizer = RobustnessOptimizer(system)
    
    # Run optimization with fewer iterations for the example
    print("\nStarting optimization (reduced iterations for example)...")
    best_k, best_K, best_g, best_s, costs, mode_gaps = optimizer.optimize(
        initial_state, num_iterations=1000, verbose=True
    )
    
    # Print summary
    print_optimization_summary(costs, mode_gaps, best_k, best_K, fixed_point)
    
    # Test robustness
    print("\nTesting robustness...")
    controlled_distances, uncontrolled_distances = optimizer.test_robustness_distribution(
        num_tests=1000  # Reduced for example
    )
    
    # Calculate improvement
    mean_controlled = np.mean(controlled_distances)
    mean_uncontrolled = np.mean(uncontrolled_distances)
    improvement = (mean_uncontrolled - mean_controlled) / mean_uncontrolled * 100
    
    print(f"Robustness test results:")
    print(f"  Mean controlled distance: {mean_controlled:.6f}")
    print(f"  Mean uncontrolled distance: {mean_uncontrolled:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    # Create summary plot
    print("\nCreating summary plot...")
    plot_optimization_summary(
        costs, mode_gaps, controlled_distances, uncontrolled_distances,
        save_path="results/example_summary.svg", show_plot=False
    )
    
    print("\nExample complete! Check results/example_summary.svg for the plot.")
    print("="*60)


def run_comparison_example():
    """
    Run a comparison between different control strategies.
    """
    print("="*60)
    print("COMPARISON EXAMPLE: Different Control Strategies")
    print("="*60)
    
    # Create the same network for fair comparison
    np.random.seed(42)  # For reproducibility
    d = config.SYSTEM_DIMENSION
    k = np.random.rand(d, d) * 0.1
    K = np.random.rand(d, d) * 0.1
    mask = np.random.rand(d, d) > 0.8
    k[~mask] = 0
    K[~mask] = 0
    
    initial_state = np.zeros(d) + 0.5
    
    strategies = ["simple", "random", "eigenmode"]
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        # Create system
        system = DynamicalSystem(k=k, K=K, lam=config.DECAY_VECTOR)
        
        # Find fixed point
        fixed_point = system.find_fixed_point(initial_state, config.SIMULATION_TIME)
        
        # Setup control based on strategy
        if strategy == "simple":
            g, s = create_simple_control_matrices(d, config.NUM_CONTROL_CHANNELS)
        elif strategy == "random":
            g = np.random.randn(d, config.NUM_CONTROL_CHANNELS)
            s = np.random.randn(config.NUM_CONTROL_CHANNELS, d)
            g /= np.linalg.norm(g, axis=0)[np.newaxis, :]
            s /= np.linalg.norm(s, axis=1)[:, np.newaxis]
        elif strategy == "eigenmode":
            _, v, _ = system.get_eigenmodes(fixed_point)
            g = v[:, :config.NUM_CONTROL_CHANNELS]
            s = v[:, :config.NUM_CONTROL_CHANNELS].T
            g /= np.linalg.norm(g, axis=0)[np.newaxis, :]
            s /= np.linalg.norm(s, axis=1)[:, np.newaxis]
        
        system.update_control_matrices(g, s)
        system.k_p = config.PROPORTIONAL_GAIN
        system.k_i = config.INTEGRAL_GAIN
        
        # Test robustness
        optimizer = RobustnessOptimizer(system)
        optimizer.current_n_f = fixed_point
        
        controlled_distances, uncontrolled_distances = optimizer.test_robustness_distribution(
            num_tests=500  # Reduced for example
        )
        
        mean_controlled = np.mean(controlled_distances)
        mean_uncontrolled = np.mean(uncontrolled_distances)
        improvement = (mean_uncontrolled - mean_controlled) / mean_uncontrolled * 100
        
        results[strategy] = {
            'controlled': mean_controlled,
            'uncontrolled': mean_uncontrolled,
            'improvement': improvement,
            'mode_gap': system.calc_mode_gap(fixed_point)
        }
        
        print(f"  Improvement: {improvement:.2f}%")
        print(f"  Mode gap: {results[strategy]['mode_gap']:.4f}")
    
    # Print comparison
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    print(f"{'Strategy':<12} {'Improvement':<12} {'Mode Gap':<10} {'Controlled':<12}")
    print("-" * 60)
    for strategy, result in results.items():
        print(f"{strategy:<12} {result['improvement']:<12.2f} {result['mode_gap']:<10.4f} {result['controlled']:<12.6f}")
    
    print("="*60)


if __name__ == "__main__":
    # Run simple example
    run_simple_example()
    
    # Run comparison example
    run_comparison_example() 