"""
Configuration file for the Duality Robustness Simulation
Contains all parameters and settings for the dynamical system simulation
"""

import numpy as np

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================

# System dimensions
SYSTEM_DIMENSION = 30
NUM_CONTROL_CHANNELS = 1

# Time parameters
SIMULATION_TIME = 2000
ROBUSTNESS_TEST_TIME = 100

# Decay parameters
DECAY_RATE = 5.0
DECAY_VECTOR = np.zeros(SYSTEM_DIMENSION) + DECAY_RATE

# Control parameters
PROPORTIONAL_GAIN = 10.0
INTEGRAL_GAIN = 1000.0

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================

# Simulated annealing parameters
INITIAL_TEMPERATURE = 0.01
COOLING_RATE = 0.9999
NUM_OPTIMIZATION_ITERATIONS = 6000

# Network perturbation parameters
NETWORK_PERTURBATION_STRENGTH = 0.4  # 2/5 as in original
CONTROL_PERTURBATION_STRENGTH = 0.2

# =============================================================================
# ROBUSTNESS TESTING PARAMETERS
# =============================================================================

# Number of perturbations to test
NUM_PERTURBATIONS = 20
NUM_ROBUSTNESS_TESTS = 10000

# Perturbation strength
PERTURBATION_STRENGTH = .1

# =============================================================================
# STABILITY CONSTRAINTS
# =============================================================================

# IPR threshold for stability
IPR_THRESHOLD = 2.0 / SYSTEM_DIMENSION

# Minimum total population
MIN_TOTAL_POPULATION = 0.2

# =============================================================================
# PLOTTING PARAMETERS
# =============================================================================

# Figure dimensions (mm to inches conversion)
MM_TO_INCHES = 1/25.4
FIGURE_WIDTH_MM = 85
FIGURE_HEIGHT_MM = 65
FIGURE_WIDTH_IN = 1.5 * FIGURE_WIDTH_MM * MM_TO_INCHES
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM * MM_TO_INCHES

# Font settings
FONT_FAMILY = 'sans-serif'
FONT_NAME = 'Helvetica'
BASE_FONT_SIZE = 9

# =============================================================================
# FILE PATHS
# =============================================================================

# Data files
KS_FILE_PATH = "many_ks.csv"
K_MATRIX_FILE_PATH = "many_KKs.csv"

# Output files
OUTPUT_PREFIX = "robustness_simulation"
MODE_GAP_PLOT_FILE = "mode_gap_evolution.svg"
ROBUSTNESS_DISTRIBUTION_PLOT_FILE = "robustness_distribution.svg"
TRADEOFF_PLOT_FILE = "robustness_tradeoff.svg"

# =============================================================================
# NUMERICAL INTEGRATION PARAMETERS
# =============================================================================

# ODE solver parameters
RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-9
NUM_TIME_POINTS = 1000 