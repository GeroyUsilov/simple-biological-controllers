# Duality Robustness Simulation

This project implements a dynamical system with integral feedback control that develops slow modes and exhibits robustness properties. The system is optimized using simulated annealing to find network structures and control parameters that maximize robustness while maintaining stability.

## Overview

The simulation demonstrates how biological systems might evolve both their connectivity and regulatory mechanisms to achieve robustness. The key insight is that there's a fundamental trade-off between robustness and mode separation - systems with better robustness tend to have smaller mode gaps.

### Key Features

- **Integral Feedback Control**: Provides persistent correction that helps maintain system stability under perturbations
- **Slow Mode Development**: The system evolves to develop a clear separation between fast and slow modes
- **Robustness Optimization**: Uses simulated annealing to find optimal network structures and control parameters
- **Dual Optimization**: Simultaneously optimizes both network structure and control parameters

## Mathematical Model

The system evolves according to:

```
dn/dt = f(n) - λn - df*n - k_p*u_p - k_i*u_i
du_i/dt = s*(n - n_f)
```

where:
- `n` is the state vector
- `u_i` is the integral control variable
- `f(n)` is the nonlinear interaction term
- `λ` is the decay rate
- `df` is external perturbation
- `k_p`, `k_i` are control gains
- `s` is the sensing matrix
- `g` is the actuation matrix

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd duality_robustness_final_code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete simulation with default parameters:

```bash
python main.py
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --network_index INT     Index of the network to use (default: 0)
  --iterations INT        Number of optimization iterations (default: from config)
  --no_plots             Disable plotting
  --control_strategy STR  Control design strategy: eigenmode, simple, random (default: eigenmode)
  --no_robustness_test   Skip robustness testing
  --output_prefix STR    Prefix for output files
```

### Examples

Run with a specific network and custom iterations:
```bash
python main.py --network_index 1 --iterations 3000
```

Run with simple control strategy and no plots:
```bash
python main.py --control_strategy simple --no_plots
```

Run with custom output prefix:
```bash
python main.py --output_prefix my_experiment
```

## Project Structure

```
duality_robustness_final_code/
├── main.py                 # Main script
├── config.py              # Configuration parameters
├── dynamical_system.py    # Dynamical system implementation
├── optimizer.py           # Simulated annealing optimizer
├── data_loader.py         # Data loading utilities
├── visualization.py       # Plotting and analysis tools
├── functions.py           # Original helper functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── results/              # Output directory (created automatically)
```

## Configuration

All parameters are centralized in `config.py`:

- **System Parameters**: Dimensions, time scales, decay rates
- **Optimization Parameters**: Temperature, cooling rate, iteration counts
- **Robustness Testing**: Number of perturbations, perturbation strength
- **Plotting Parameters**: Figure sizes, fonts, output paths

## Output

The simulation generates several types of output:

### Files
- **Matrices**: Optimized k, K, g, s matrices
- **History**: Cost and mode gap evolution
- **Robustness**: Controlled vs uncontrolled distance distributions

### Plots
- **Mode Gap Evolution**: Shows how the mode gap changes during optimization
- **Robustness Distribution**: Compares controlled vs uncontrolled system responses
- **Robustness Tradeoff**: Scatter plot of mode gap vs cost
- **Summary**: Comprehensive overview of all results

### Console Output
- **Setup Information**: System parameters and initial conditions
- **Optimization Progress**: Real-time cost and mode gap updates
- **Final Summary**: Key statistics and improvements achieved

## Understanding the Results

### Mode Gap
The mode gap is the ratio of the second slowest to the slowest eigenvalue. A larger mode gap indicates better separation between fast and slow dynamics.

### Robustness Cost
The robustness cost measures how well the system maintains its fixed point under perturbations. Lower values indicate better robustness.

### Trade-off
There's typically a trade-off between mode gap and robustness - systems with better robustness tend to have smaller mode gaps, suggesting that the development of slow modes comes at the cost of some robustness.

## Data Requirements

The simulation expects network data files:
- `../many_ks.csv`: Interaction strength matrices
- `../many_KKs.csv`: Interaction matrices

If these files are not available, the system will automatically generate random test data.

## Troubleshooting

### Common Issues

1. **Missing Data Files**: The system will automatically use random data if the CSV files are not found.

2. **Memory Issues**: For large systems, try reducing the number of iterations or system dimension in `config.py`.

3. **Slow Performance**: The optimization can be computationally intensive. Consider using `--no_plots` for faster runs.

4. **Convergence Issues**: If the optimization doesn't converge well, try adjusting the temperature and cooling rate in `config.py`.

### Error Messages

- **"Network data validation failed"**: The loaded data doesn't meet expected format requirements
- **"Optimization failed to converge"**: The simulated annealing didn't find good solutions
- **"System became unstable"**: The optimization produced an unstable system

## Extending the Code

### Adding New Control Strategies

1. Add the strategy to `setup_control()` in `main.py`
2. Implement the control matrix generation logic
3. Update the argument parser choices

### Modifying the Objective Function

1. Edit the `evaluate_robustness()` method in `DynamicalSystem`
2. Adjust the perturbation testing logic
3. Update the normalization if needed

### Adding New Analysis Tools

1. Create new plotting functions in `visualization.py`
2. Add them to the main workflow in `main.py`
3. Update the output saving logic if needed

## Citation

If you use this code in your research, please cite the original work:

```
[Add citation information here]
```

## License

[Add license information here]

## Contributing

[Add contribution guidelines here] 