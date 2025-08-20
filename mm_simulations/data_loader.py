"""
Data Loader for Network Parameters

This module handles loading network interaction matrices from CSV files
and provides utilities for data preprocessing.
"""

import numpy as np
from typing import List, Tuple
import config


def load_3d_matrix_from_csv(csv_file_path: str) -> np.ndarray:
    """
    Load a 3D matrix from a CSV file.
    
    The CSV file is expected to have the following format:
    - Empty lines or lines starting with "Depth" separate different 2D matrices
    - Each non-empty line contains comma-separated values for one row of a 2D matrix
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        3D numpy array containing the matrices
    """
    matrix_slices = []
    current_slice = []
    
    try:
        with open(csv_file_path, 'r') as csv_file:
            for line in csv_file:
                if line.strip():  # Non-empty line
                    if line.startswith("Depth"):
                        if current_slice:
                            matrix_slices.append(current_slice)
                            current_slice = []
                    else:
                        values = line.strip().split(',')
                        current_slice.append(list(map(float, values)))
        
        if current_slice:
            matrix_slices.append(current_slice)
        
        matrix_3d = np.array(matrix_slices)
        return matrix_3d
        
    except FileNotFoundError:
        print(f"Warning: File {csv_file_path} not found. Creating random matrices.")
        return _create_random_matrices()
    except Exception as e:
        print(f"Error loading {csv_file_path}: {e}")
        return _create_random_matrices()


def _create_random_matrices() -> np.ndarray:
    """
    Create random matrices for testing when data files are not available.
    
    Returns:
        3D numpy array with random interaction matrices
    """
    d = config.SYSTEM_DIMENSION
    # Create a few random matrices for testing
    matrices = []
    for i in range(3):  # Create 3 random matrices
        # Create sparse random matrix
        k = np.random.rand(d, d) * 0.1
        K = np.random.rand(d, d) * 0.1
        
        # Ensure some sparsity
        mask = np.random.rand(d, d) > 0.7
        k[~mask] = 0
        K[~mask] = 0
        
        matrices.append(k)
    
    return np.array(matrices)


def load_network_data(ks_file_path: str = None, Ks_file_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load network interaction data from CSV files.
    
    Args:
        ks_file_path: Path to the k matrices file
        Ks_file_path: Path to the K matrices file
        
    Returns:
        Tuple of (ks_matrices, Ks_matrices)
    """
    ks_file_path = ks_file_path or config.KS_FILE_PATH
    Ks_file_path = Ks_file_path or config.K_MATRIX_FILE_PATH
    
    print(f"Loading k matrices from: {ks_file_path}")
    ks = load_3d_matrix_from_csv(ks_file_path)
    
    print(f"Loading K matrices from: {Ks_file_path}")
    Ks = load_3d_matrix_from_csv(Ks_file_path)
    
    print(f"Loaded {len(ks)} k matrices and {len(Ks)} K matrices")
    print(f"Matrix dimensions: {ks[0].shape}")
    
    return ks, Ks


def get_network_pair(ks: np.ndarray, Ks: np.ndarray, index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a specific pair of k and K matrices.
    
    Args:
        ks: Array of k matrices
        Ks: Array of K matrices
        index: Index of the matrices to retrieve
        
    Returns:
        Tuple of (k_matrix, K_matrix)
    """
    if index >= len(ks) or index >= len(Ks):
        raise ValueError(f"Index {index} out of range. Available matrices: {min(len(ks), len(Ks))}")
    
    return ks[index], Ks[index]


def validate_network_data(ks: np.ndarray, Ks: np.ndarray) -> bool:
    """
    Validate that the loaded network data is consistent.
    
    Args:
        ks: Array of k matrices
        Ks: Array of K matrices
        
    Returns:
        True if data is valid, False otherwise
    """
    if len(ks) != len(Ks):
        print(f"Warning: Number of k matrices ({len(ks)}) != number of K matrices ({len(Ks)})")
        return False
    
    if len(ks) == 0:
        print("Warning: No matrices loaded")
        return False
    
    # Check that all matrices have the same shape
    expected_shape = ks[0].shape
    for i, k in enumerate(ks):
        if k.shape != expected_shape:
            print(f"Warning: k matrix {i} has shape {k.shape}, expected {expected_shape}")
            return False
    
    for i, K in enumerate(Ks):
        if K.shape != expected_shape:
            print(f"Warning: K matrix {i} has shape {K.shape}, expected {expected_shape}")
            return False
    
    print(f"Network data validation passed: {len(ks)} matrices of shape {expected_shape}")
    return True


def create_control_matrices_from_eigenmodes(v: np.ndarray, num_channels: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create control matrices based on the system's eigenmodes.
    
    Args:
        v: Eigenvector matrix from eigendecomposition
        num_channels: Number of control channels (defaults to system dimension)
        
    Returns:
        Tuple of (g_matrix, s_matrix)
    """
    d = v.shape[0]
    num_channels = num_channels or d
    
    # Use the first num_channels eigenvectors
    g = v[:, :num_channels]
    s = v[:, :num_channels].T
    
    # Normalize
    g /= np.linalg.norm(g, axis=0)[np.newaxis, :]
    s /= np.linalg.norm(s, axis=1)[:, np.newaxis]
    
    return g, s


def create_simple_control_matrices(d: int, num_channels: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create simple control matrices for testing.
    
    Args:
        d: System dimension
        num_channels: Number of control channels
        
    Returns:
        Tuple of (g_matrix, s_matrix)
    """
    g = np.zeros((d, num_channels))
    s = np.zeros((num_channels, d))
    
    # Simple diagonal control
    for i in range(min(d, num_channels)):
        g[i, i] = 1.0
        s[i, i] = 1.0
    
    # Normalize
    g /= np.linalg.norm(g, axis=0)[np.newaxis, :]
    s /= np.linalg.norm(s, axis=1)[:, np.newaxis]
    
    return g, s 