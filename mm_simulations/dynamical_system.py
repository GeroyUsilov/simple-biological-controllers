"""
Dynamical System with Integral Feedback Control

This module implements a dynamical system with integral feedback control
that can develop slow modes and exhibit robustness properties.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
import config


class DynamicalSystem:
    """
    A dynamical system with integral feedback control.
    
    The system evolves according to:
    dn/dt = f(n) - λn - df*n - k_p*u_p - k_i*u_i
    du_i/dt = s*(n - n_f)
    
    where:
    - n is the state vector
    - u_i is the integral control variable
    - f(n) is the nonlinear interaction term
    - λ is the decay rate
    - df is external perturbation
    - k_p, k_i are control gains
    - s is the sensing matrix
    - g is the actuation matrix
    """
    
    def __init__(self, k: np.ndarray, K: np.ndarray, lam: np.ndarray, 
                 k_p: float = 0.0, k_i: float = 0.0, 
                 g: Optional[np.ndarray] = None, 
                 s: Optional[np.ndarray] = None):
        """
        Initialize the dynamical system.
        
        Args:
            k: Interaction strength matrix
            K: Interaction matrix
            lam: Decay rate vector
            k_p: Proportional control gain
            k_i: Integral control gain
            g: Actuation matrix (optional)
            s: Sensing matrix (optional)
        """
        self.k = k
        self.K = K
        self.lam = lam
        self.k_p = k_p
        self.k_i = k_i
        self.d = k.shape[0]
        
        # Initialize control matrices if not provided
        if g is None:
            self.g = np.zeros((self.d, 1))
        else:
            self.g = g
            
        if s is None:
            self.s = np.zeros((1, self.d))
        else:
            self.s = s
            
        self.l = self.g.shape[1]  # Number of control channels
        
    def calc_dydt_controlled(self, y: np.ndarray, df: np.ndarray, 
                           n_f: np.ndarray) -> np.ndarray:
        """
        Calculate the time derivative of the controlled system.
        
        Args:
            y: Combined state vector [n, u_i]
            df: External perturbation vector
            n_f: Target fixed point
            
        Returns:
            Time derivative of the combined state
        """
        n = y[0:self.d]
        u_i = y[self.d:self.d + self.l]
        
        # Proportional control signal
        u_p = np.matmul(self.s, n - n_f)
        
        # State dynamics
        interaction_term = np.matmul(
            self.k * (np.outer(np.ones(len(n)), n) + self.K)**(-1), n
        )
        dndt = (interaction_term - self.lam * n - df * n - 
                self.k_p * np.matmul(self.g, u_p) - 
                self.k_i * np.matmul(self.g, u_i))
        
        # Integral control dynamics
        duidt = np.matmul(self.s, n - n_f)
        
        # Combine derivatives
        dydt = np.zeros(self.d + self.l)
        dydt[0:self.d] = dndt
        dydt[self.d:self.d + self.l] = duidt
        
        return dydt
    
    def calc_dydt_uncontrolled(self, y: np.ndarray, df: np.ndarray) -> np.ndarray:
        """
        Calculate the time derivative of the uncontrolled system.
        
        Args:
            y: Combined state vector [n, u_i]
            df: External perturbation vector
            
        Returns:
            Time derivative of the combined state
        """
        n = y[0:self.d]
        u_i = y[self.d:self.d + self.l]
        
        # State dynamics (no control terms)
        interaction_term = np.matmul(
            self.k * (np.outer(np.ones(len(n)), n) + self.K)**(-1), n
        )
        dndt = interaction_term - self.lam * n - df * n
        
        # Integral control dynamics (still evolve but don't affect state)
        duidt = np.zeros(self.l)
        
        # Combine derivatives
        dydt = np.zeros(self.d + self.l)
        dydt[0:self.d] = dndt
        dydt[self.d:self.d + self.l] = duidt
        
        return dydt
    
    def simulate(self, n_0: np.ndarray, u_i_0: np.ndarray, t_f: float,
                df: np.ndarray, n_f: np.ndarray, use_control: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the dynamical system.
        
        Args:
            n_0: Initial state
            u_i_0: Initial integral control
            t_f: Final time
            df: External perturbation
            n_f: Target fixed point
            use_control: Whether to use control (default: True)
            
        Returns:
            Tuple of (state evolution, integral control evolution, time points)
        """
        def dydt(t, y):
            if use_control:
                return self.calc_dydt_controlled(y, df, n_f)
            else:
                return self.calc_dydt_uncontrolled(y, df)
        
        # Set up initial conditions
        y_0 = np.zeros(self.d + self.l)
        y_0[0:self.d] = n_0
        y_0[self.d:self.d + self.l] = u_i_0
        
        # Solve ODE
        t_span = [0, t_f]
        sol = solve_ivp(
            dydt, t_span, y_0,
            rtol=config.RELATIVE_TOLERANCE,
            atol=config.ABSOLUTE_TOLERANCE,
            dense_output=True,
            method='Radau'
        )
        
        # Evaluate solution at regular time points
        t_eval = np.linspace(t_span[0], t_span[1], config.NUM_TIME_POINTS)
        y_eval = sol.sol(t_eval)
        
        n_eval = y_eval[0:self.d, :]
        u_i_eval = y_eval[self.d:self.d + self.l, :]
        
        return n_eval, u_i_eval, t_eval
    
    def find_fixed_point(self, n_0: np.ndarray, t_f: float) -> np.ndarray:
        """
        Find the fixed point of the uncontrolled system.
        
        Args:
            n_0: Initial guess
            t_f: Simulation time
            
        Returns:
            Fixed point state
        """
        # Simulate without control to find fixed point
        n_eval, _, _ = self.simulate(
            n_0, np.zeros(self.l), t_f, 
            np.zeros(self.d), n_0, use_control=False
        )
        return n_eval[:, -1]
    
    def calc_jacobian(self, n: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian matrix at state n.
        
        Args:
            n: State vector
            
        Returns:
            Jacobian matrix
        """
        return (self.k * self.K / 
                (self.K**2 + 2 * self.K * np.outer(np.ones(len(n)), n) + 
                 np.outer(np.ones(len(n)), n)**2) - 
                self.lam * np.identity(len(n)))
    
    def get_eigenmodes(self, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate eigenvalues and eigenvectors at state n.
        
        Args:
            n: State vector
            
        Returns:
            Tuple of (eigenvalues, eigenvectors, real parts of eigenvalues)
        """
        J = self.calc_jacobian(n)
        w_c, v = np.linalg.eig(J)
        w = np.real(w_c)
        return w_c, v, w
    
    def calc_mode_gap(self, n: np.ndarray) -> float:
        """
        Calculate the mode gap (ratio of second slowest to slowest eigenvalue).
        
        Args:
            n: State vector
            
        Returns:
            Mode gap ratio
        """
        _, _, w = self.get_eigenmodes(n)
        sorted_eigenvalues = np.sort(w)
        return sorted_eigenvalues[-2] / sorted_eigenvalues[-1]
    
    def calc_ipr(self, n: np.ndarray) -> float:
        """
        Calculate the Inverse Participation Ratio for stability analysis.
        
        Args:
            n: State vector
            
        Returns:
            IPR value
        """
        w_c, v, w = self.get_eigenmodes(n)
        # IPR of the slowest mode
        slowest_mode_idx = np.argsort(w)[-1]
        v_slowest = v[:, slowest_mode_idx]
        return np.sum((v_slowest * np.conjugate(v_slowest))**2)
    
    def evaluate_robustness(self, n_f: np.ndarray, m: int, t_f: float) -> float:
        """
        Evaluate system robustness by testing perturbations.
        
        Args:
            n_f: Fixed point
            m: Number of perturbations to test
            t_f: Simulation time
            
        Returns:
            Normalized robustness measure (lower is better)
        """
        dists_controlled = []
        dists_uncontrolled = []
        
        for i in range(m):
            # Apply perturbation to component i
            df = np.zeros(self.d)
            df[i] = config.PERTURBATION_STRENGTH
            
            # Test with control
            nt, _, _ = self.simulate(
                n_f, np.zeros(self.l), t_f, df, n_f, use_control=True
            )
            dists_controlled.append(self.distance(n_f, nt[:, -1]))
            
            # Test without control
            nt, _, _ = self.simulate(
                n_f, np.zeros(self.l), t_f, df, n_f, use_control=False
            )
            dists_uncontrolled.append(self.distance(n_f, nt[:, -1]))
        
        return np.mean(dists_controlled) / np.mean(dists_uncontrolled)
    
    def distance(self, n_f: np.ndarray, n_p: np.ndarray) -> float:
        """
        Calculate normalized distance between two states.
        
        Args:
            n_f: Reference state
            n_p: Perturbed state
            
        Returns:
            Normalized distance
        """
        return np.linalg.norm(n_f - n_p) / np.linalg.norm(n_f)
    
    def update_control_matrices(self, g: np.ndarray, s: np.ndarray):
        """
        Update the control matrices.
        
        Args:
            g: New actuation matrix
            s: New sensing matrix
        """
        self.g = g
        self.s = s
        self.l = g.shape[1]
        
    def normalize_control_matrices(self):
        """
        Normalize control matrices to unit norm.
        """
        self.g /= np.linalg.norm(self.g, axis=0)[np.newaxis, :]
        self.s /= np.linalg.norm(self.s, axis=1)[:, np.newaxis] 