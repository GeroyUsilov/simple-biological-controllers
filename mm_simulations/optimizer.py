"""
Optimizer for Dynamical System Robustness

This module implements simulated annealing optimization to find
network structures and control parameters that maximize robustness
while developing slow modes.
"""

import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Optional
import config
from dynamical_system import DynamicalSystem


class RobustnessOptimizer:
    """
    Optimizer that uses simulated annealing to find robust network structures
    and control parameters that develop slow modes.
    """
    
    def __init__(self, system: DynamicalSystem, temperature: float = None, 
                 cooling_rate: float = None):
        """
        Initialize the optimizer.
        
        Args:
            system: Dynamical system to optimize
            temperature: Initial temperature for simulated annealing
            cooling_rate: Cooling rate for temperature schedule
        """
        self.system = system
        self.temperature = temperature or config.INITIAL_TEMPERATURE
        self.cooling_rate = cooling_rate or config.COOLING_RATE
        
        # Store best solutions found
        self.best_k = np.array(system.k)
        self.best_K = np.array(system.K)
        self.best_g = np.array(system.g)
        self.best_s = np.array(system.s)
        self.best_n_f = None
        self.best_cost = float('inf')
        
        # Current solutions
        self.current_k = np.array(system.k)
        self.current_K = np.array(system.K)
        self.current_g = np.array(system.g)
        self.current_s = np.array(system.s)
        self.current_n_f = None
        self.current_cost = float('inf')
        self.current_mode_gap = 0.0
        
        # Optimization history
        self.costs = []
        self.mode_gaps = []
        
    def optimize(self, n_0: np.ndarray, num_iterations: int = None, 
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                             np.ndarray, List[float], List[float]]:
        """
        Run the optimization process.
        
        Args:
            n_0: Initial state for finding fixed points
            num_iterations: Number of optimization iterations
            verbose: Whether to show progress bar
            
        Returns:
            Tuple of (best_k, best_K, best_g, best_s, costs, mode_gaps)
        """
        num_iterations = num_iterations or config.NUM_OPTIMIZATION_ITERATIONS
        
        # Initialize fixed point and cost
        self.current_n_f = self.system.find_fixed_point(n_0, config.SIMULATION_TIME)
        self.current_cost = self.system.evaluate_robustness(
            self.current_n_f, config.NUM_PERTURBATIONS, config.ROBUSTNESS_TEST_TIME
        )
        self.current_mode_gap = self.system.calc_mode_gap(self.current_n_f)
        
        # Initialize best solution
        self.best_n_f = np.array(self.current_n_f)
        self.best_cost = self.current_cost
        
        # Get connected edges for network perturbations
        is_connected = self.current_k != 0
        
        # Run optimization
        iterator = tqdm(range(num_iterations)) if verbose else range(num_iterations)
        
        for iteration in iterator:
            # Store current values
            self.costs.append(self.current_cost)
            self.mode_gaps.append(self.current_mode_gap)
            
            # Alternate between network and control optimization
            if iteration % 1 == 0:
                self._optimize_network_structure(n_0, is_connected)
            else:
                self._optimize_control_parameters()
            
            # Update temperature
            self.temperature *= self.cooling_rate
            
            # Update progress bar
            if verbose:
                iterator.set_description(
                    f"Cost: {self.current_cost:.4f}, "
                    f"Mode Gap: {self.current_mode_gap:.4f}, "
                    f"Temp: {self.temperature:.6f}"
                )
        
        return (self.best_k, self.best_K, self.best_g, self.best_s, 
                self.costs, self.mode_gaps)
    
    def _optimize_network_structure(self, n_0: np.ndarray, is_connected: np.ndarray):
        """
        Optimize the network structure (k and K matrices).
        
        Args:
            n_0: Initial state for finding fixed points
            is_connected: Boolean matrix indicating connected edges
        """
        # Select random connected edge
        connected_edges = np.nonzero(is_connected)
        if len(connected_edges[0]) == 0:
            return
            
        idx = np.random.randint(len(connected_edges[0]))
        i, j = connected_edges[0][idx], connected_edges[1][idx]
        
        # Propose new network parameters
        k_new = np.array(self.current_k)
        K_new = np.array(self.current_K)
        
        k_new[i, j] += (np.random.rand() - 0.5) * config.NETWORK_PERTURBATION_STRENGTH
        K_new[i, j] += (np.random.rand() - 0.5) * config.NETWORK_PERTURBATION_STRENGTH
        
        # Check bounds
        if (k_new[i, j] < 1 and K_new[i, j] < 1 and 
            K_new[i, j] > 0 and k_new[i, j] > 0):
            
            # Create temporary system with new parameters
            temp_system = DynamicalSystem(
                k_new, K_new, self.system.lam,
                self.system.k_p, self.system.k_i,
                self.current_g, self.current_s
            )
            
            # Find new fixed point
            n_f_new = temp_system.find_fixed_point(n_0, config.SIMULATION_TIME)
            
            # Calculate stability properties
            mode_gap_new = temp_system.calc_mode_gap(n_f_new)
            ipr_new = temp_system.calc_ipr(n_f_new)
            cost_new = temp_system.evaluate_robustness(
                n_f_new, config.NUM_PERTURBATIONS, config.ROBUSTNESS_TEST_TIME
            )
            
            # Check stability constraints
            if (ipr_new < config.IPR_THRESHOLD and 
                np.sort(temp_system.get_eigenmodes(n_f_new)[2])[-1] < 0 and
                np.sum(n_f_new) > config.MIN_TOTAL_POPULATION):
                
                # Accept or reject based on Metropolis criterion
                if cost_new < self.current_cost:
                    self._accept_network_update(k_new, K_new, n_f_new, cost_new, mode_gap_new)
                else:
                    probability = np.exp((self.current_cost - cost_new) / self.temperature)
                    if np.random.random() < probability:
                        self._accept_network_update(k_new, K_new, n_f_new, cost_new, mode_gap_new)
    
    def _optimize_control_parameters(self):
        """
        Optimize the control parameters (g and s matrices).
        """
        # Propose new control parameters
        new_s = np.array(self.current_s)
        new_g = np.array(self.current_g)
        
        # Random perturbations
        i_s, j_s = np.random.randint(new_s.shape[0]), np.random.randint(new_s.shape[1])
        i_g, j_g = np.random.randint(new_g.shape[0]), np.random.randint(new_g.shape[1])
        
        new_s[i_s, j_s] += np.random.normal(0, config.CONTROL_PERTURBATION_STRENGTH)
        new_g[i_g, j_g] += np.random.normal(0, config.CONTROL_PERTURBATION_STRENGTH)
        
        # Normalize
        new_s /= np.linalg.norm(new_s, axis=1)[:, np.newaxis]
        new_g /= np.linalg.norm(new_g, axis=0)[np.newaxis, :]
        
        # Update system with new control parameters
        self.system.update_control_matrices(new_g, new_s)
        
        # Evaluate new cost
        cost_new = self.system.evaluate_robustness(
            self.current_n_f, config.NUM_PERTURBATIONS, config.ROBUSTNESS_TEST_TIME
        )
        
        # Accept or reject
        if cost_new < self.current_cost:
            self._accept_control_update(new_g, new_s, cost_new)
        else:
            probability = np.exp((self.current_cost - cost_new) / self.temperature)
            if np.random.random() < probability:
                self._accept_control_update(new_g, new_s, cost_new)
    
    def _accept_network_update(self, k_new: np.ndarray, K_new: np.ndarray, 
                             n_f_new: np.ndarray, cost_new: float, mode_gap_new: float):
        """
        Accept a network structure update.
        
        Args:
            k_new: New interaction strength matrix
            K_new: New interaction matrix
            n_f_new: New fixed point
            cost_new: New cost value
            mode_gap_new: New mode gap
        """
        self.current_k = np.array(k_new)
        self.current_K = np.array(K_new)
        self.current_n_f = np.array(n_f_new)
        self.current_cost = cost_new
        self.current_mode_gap = mode_gap_new
        
        # Update system
        self.system.k = k_new
        self.system.K = K_new
        
        # Update best solution if better
        if cost_new < self.best_cost:
            self.best_k = np.array(k_new)
            self.best_K = np.array(K_new)
            self.best_n_f = np.array(n_f_new)
            self.best_cost = float(cost_new)
    
    def _accept_control_update(self, g_new: np.ndarray, s_new: np.ndarray, cost_new: float):
        """
        Accept a control parameter update.
        
        Args:
            g_new: New actuation matrix
            s_new: New sensing matrix
            cost_new: New cost value
        """
        self.current_g = np.array(g_new)
        self.current_s = np.array(s_new)
        self.current_cost = cost_new
        
        # Update best solution if better
        if cost_new < self.best_cost:
            self.best_g = np.array(g_new)
            self.best_s = np.array(s_new)
            self.best_cost = cost_new
    
    def test_robustness_distribution(self, num_tests: int = None) -> Tuple[List[float], List[float]]:
        """
        Test robustness by applying random perturbations and comparing
        controlled vs uncontrolled responses.
        
        Args:
            num_tests: Number of tests to run
            
        Returns:
            Tuple of (controlled_distances, uncontrolled_distances)
        """
        num_tests = num_tests or config.NUM_ROBUSTNESS_TESTS
        
        controlled_distances = []
        uncontrolled_distances = []
        
        # Get connected edges
        is_connected = self.current_k != 0
        
        for _ in tqdm(range(num_tests), desc="Testing robustness"):
            # Select random connected edge
            connected_edges = np.nonzero(is_connected)
            if len(connected_edges[0]) == 0:
                continue
                
            idx = np.random.randint(len(connected_edges[0]))
            i, j = connected_edges[0][idx], connected_edges[1][idx]
            
            # Create perturbed system
            k_perturbed = np.array(self.current_k)
            K_perturbed = np.array(self.current_K)
            k_perturbed[i, j] += (np.random.rand() - 0.5)
            K_perturbed[i, j] += (np.random.rand() - 0.5)
            
            # Test with control
            temp_system = DynamicalSystem(
                k_perturbed, K_perturbed, self.system.lam,
                self.system.k_p, self.system.k_i,
                self.current_g, self.current_s
            )
            nt, _, _ = temp_system.simulate(
                self.current_n_f, np.zeros(self.system.l), 
                config.ROBUSTNESS_TEST_TIME, np.zeros(self.system.d), self.current_n_f
            )
            controlled_distances.append(self.system.distance(self.current_n_f, nt[:, -1]))
            
            # Test without control
            temp_system_no_control = DynamicalSystem(
                k_perturbed, K_perturbed, self.system.lam,
                0, 0, np.zeros(self.current_g.shape), np.zeros(self.current_s.shape)
            )
            nt, _, _ = temp_system_no_control.simulate(
                self.current_n_f, np.zeros(self.system.l), 
                config.ROBUSTNESS_TEST_TIME, np.zeros(self.system.d), self.current_n_f
            )
            uncontrolled_distances.append(self.system.distance(self.current_n_f, nt[:, -1]))
        
        return controlled_distances, uncontrolled_distances 