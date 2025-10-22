# Running Fitted Q-Iteration on Maze Environments

This README explains how to use the `run_fqi_maze.py` script to train a tabular Q-learning agent using Fitted Q-Iteration (FQI) on maze/grid environments and visualize the results.

## Overview

This script provides a complete, self-contained example of running FQI on maze environments with comprehensive visualization. It demonstrates:
- Creating maze environments from string specifications
- Running exact FQI with a tabular neural network
- Visualizing Q-values and learned policies
- Plotting learning curves and performance metrics
- Saving Q-values for further analysis

### Key Features

**1. Environment Creation**
- Creates tabular maze environments from simple string specifications
- Uses the Cython-optimized `grid_env_cy` from this repository
- Supports various tile types: start, reward, walls, lava, etc.

**2. FQI Training**
- Implements exact FQI using the `ExactFQI` class from `debugq.algos`
- Uses tabular Q-networks from `debugq.models.q_networks`
- Includes early stopping with both absolute and relative tolerance
- Achieves ~99.9% of optimal performance on test mazes

**3. Comprehensive Visualization**
The script generates 5 different PDF visualizations:
- **Learning curves** (6 subplots): returns, normalized returns, TD error, Q-value evolution, Bellman error, gradient steps
- **Q-value heatmaps**: Optimal Q-values (ground truth) and learned Q-values with color-coded triangles for each action
- **Policy visualizations**: Learned and optimal policies with probability-weighted arrows

**4. Data Saving**
- Saves learned and optimal Q-values as NumPy arrays (`.npy` and `.npz` formats)
- Organized output structure with maze-specific folders
- Publication-ready PDFs with tight bounding boxes

## Requirements

The script uses modules from this repository. Make sure you've built the Cython extensions:

```bash
make build
```

## Quick Start

Run the script with default settings:

```bash
python run_fqi_maze.py
```

This will:
1. Create a 6x4 maze environment
2. Train FQI for 50 iterations
3. Save results to `fqi_results/maze_6x4/`
4. Exit without opening any matplotlib windows

## Output Structure

Results are organized by maze name in the following structure:

```
fqi_results/
└── maze_6x4/
    ├── q_values.npz              # All Q-values and metrics
    ├── learned_q.npy             # Learned Q-values array
    ├── optimal_q.npy             # Optimal Q-values array
    ├── learning_curves.pdf       # Training progress (6 subplots)
    ├── qstar.pdf                 # Optimal Q-value visualization
    ├── learned_q.pdf             # Learned Q-value visualization
    ├── policy.pdf                # Learned policy
    └── optimal_policy.pdf        # Optimal policy
```

### Output Files

**Q-values (NumPy arrays):**
- **`q_values.npz`**: Comprehensive file containing:
  - `learned_q`: Final learned Q-values
  - `optimal_q`: Optimal Q-values (Q*)
  - `expert_returns`: Performance of optimal policy
  - `random_returns`: Performance of random policy
  - `final_returns`: Final agent performance
  - `final_normalized_returns`: Normalized final performance
  
- **`learned_q.npy`**: Just the learned Q-values (shape: `[num_states, num_actions]`)
- **`optimal_q.npy`**: Just the optimal Q-values (shape: `[num_states, num_actions]`)

**Visualizations (PDFs with tight bounding boxes):**
- **`learning_curves.pdf`**: Six subplots showing:
  - Returns over iterations
  - Normalized returns (0 = random, 1 = expert)
  - Projection loss (TD error)
  - Q-value evolution
  - Bellman error to optimal Q*
  - Gradient steps per iteration

- **`qstar.pdf`**: Optimal Q-values for all state-action pairs
- **`learned_q.pdf`**: Learned Q-values after training
- **`policy.pdf`**: Learned policy visualization with arrows
- **`optimal_policy.pdf`**: Optimal policy for comparison

## Loading Saved Q-values

**Load everything (recommended):**
```python
import numpy as np

data = np.load('fqi_results/maze_6x4/q_values.npz')
learned_q = data['learned_q']
optimal_q = data['optimal_q']
expert_returns = data['expert_returns']
final_returns = data['final_returns']
```

**Load individual arrays:**
```python
learned_q = np.load('fqi_results/maze_6x4/learned_q.npy')
optimal_q = np.load('fqi_results/maze_6x4/optimal_q.npy')
```

The Q-value arrays have shape `(num_states, num_actions)`. Index with `learned_q[state, action]` to get the Q-value for that state-action pair.

## Customizing the Maze

### Using Predefined Mazes

The script includes several example mazes in the `main()` function:

```python
# Simple 3x2 maze
maze_string = "SOO\\OOR\\"

# Larger 6x4 maze (default)
maze_string = "SOOOOO\\OOOOOO\\OOOOOO\\OOOOOR\\"

# Maze with walls
maze_string = "SOOO\\O##O\\O##O\\OOOR\\"

# Maze with lava
maze_string = "OOOOOOR\\SOLLLLL\\OOOOOOO\\OOOOOO3\\"
```

### Creating Custom Mazes

Mazes are defined using string specifications where each character represents a tile type:

- `'S'` - Start position
- `'R'` - Reward (+1.0)
- `'O'` - Open space (empty)
- `'#'` - Wall (impassable)
- `'L'` - Lava (-1.0 reward, agent gets stuck)
- `'2'` - Reward level 2 (+2.0)
- `'3'` - Reward level 3 (+4.0)
- `'4'` - Reward level 4 (+8.0)

Rows are separated by `\\` (backslash). For example:

```python
maze_string = "S##\\O#R\\"
```

Creates:
```
S # #
O # R
```

### Running Multiple Mazes

```python
# Define different mazes with descriptive names
mazes = [
    ("maze_6x4", "SOOOOO\\OOOOOO\\OOOOOO\\OOOOOR\\"),
    ("maze_with_walls", "SOOO\\O##O\\O##O\\OOOR\\"),
    ("maze_with_lava", "OOOOOOR\\SOLLLLL\\OOOOOOO\\OOOOOO3\\"),
]

for maze_name, maze_string in mazes:
    fqi, log_dict, gridspec = run_fqi_on_maze(
        maze_string=maze_string,
        maze_name=maze_name,
        output_dir="fqi_results",
        num_iterations=50,
    )
```

This creates separate folders for each maze: `fqi_results/maze_6x4/`, `fqi_results/maze_with_walls/`, etc.

## Hyperparameter Tuning

Modify the `run_fqi_on_maze()` function call to adjust training parameters:

```python
fqi, log_dict, gridspec = run_fqi_on_maze(
    maze_string=maze_string,
    maze_name="my_maze",          # Name for output folder
    output_dir="fqi_results",      # Base output directory
    num_iterations=50,             # Number of FQI iterations
    time_limit=50,                 # Max steps per episode
    discount=0.95,                 # Discount factor (gamma)
    ent_wt=0.1,                    # Entropy weight for soft Q-learning
    lr=5e-3,                       # Learning rate for projection
    min_project_steps=10,          # Min gradient steps per projection
    max_project_steps=100,         # Max gradient steps per projection
)
```

### Key Parameters

- **`num_iterations`**: More iterations allow better convergence but take longer
- **`discount`**: Lower values (e.g., 0.9) focus on short-term rewards; higher values (e.g., 0.99) consider long-term rewards
- **`ent_wt`**: Entropy regularization:
  - 0.0 = hard max (deterministic policy)
  - Higher values = softer, more exploratory policy
- **`lr`**: Learning rate for gradient descent:
  - Too high: unstable training
  - Too low: slow convergence
- **`min/max_project_steps`**: Controls how long to fit the Q-network at each iteration

## Understanding the Output

### Console Output

During training, you'll see progress like:

```
================================================================================
Running Fitted Q-Iteration on Maze Environment
================================================================================

1. Creating maze environment...
   - Grid size: 6 x 4
   - Num states: 24
   - Num actions: 5

2. Creating tabular Q-network...
   - Network parameters: 125

3. Setting up FQI algorithm...
   - Discount: 0.95
   - Entropy weight: 0.1
   - Learning rate: 0.005
   - Expert returns: 41.996
   - Random returns: 0.732

4. Training FQI for 50 iterations...
   Iter   1: Return=  0.460 (norm=-0.012), Loss=1.9126e-02, Steps=99
   Iter   5: Return= 10.700 (norm=0.238), Loss=9.1041e-03, Steps=99
   Iter  10: Return= 26.620 (norm=0.628), Loss=7.5682e-03, Steps=25
   ...
   Iter  50: Return= 41.920 (norm=0.999), Loss=9.2329e-03, Steps=10

5. Saving results...
   - Output directory: fqi_results/maze_6x4/
   - Saving Q-values...
   - Generating learning curves...
   - Visualizing optimal Q-values...
   - Visualizing learned Q-values...
   - Visualizing learned policy...
   - Visualizing optimal policy...

================================================================================
Training complete! Results saved to:

Q-values:
  - fqi_results/maze_6x4/q_values.npz: All Q-values and metrics
  - fqi_results/maze_6x4/learned_q.npy: Learned Q-values
  - fqi_results/maze_6x4/optimal_q.npy: Optimal Q-values

Visualizations:
  - fqi_results/maze_6x4/learning_curves.pdf: Learning progress
  - fqi_results/maze_6x4/qstar.pdf: Optimal Q-values
  - fqi_results/maze_6x4/learned_q.pdf: Learned Q-values
  - fqi_results/maze_6x4/policy.pdf: Learned policy
  - fqi_results/maze_6x4/optimal_policy.pdf: Optimal policy
================================================================================

Final Statistics:
  Expert returns: 41.996
  Random returns: 0.732
  Final returns: 41.920
  Final normalized returns: 0.999
  Final Bellman error: 6.9503e+00
```

- **Return**: Average episode return
- **norm**: Normalized return (0 = random policy, 1 = optimal policy)
- **Loss**: TD error / projection loss
- **Steps**: Number of gradient steps taken in this iteration

### Final Statistics

```
Final Statistics:
  Expert returns: 41.996        # Optimal policy performance
  Random returns: 0.732         # Random policy baseline
  Final returns: 41.920         # Your agent's performance
  Final normalized returns: 0.999  # 99.9% of optimal!
  Final Bellman error: 6.9503e+00  # Distance from optimal Q*
```

## Visualization Details

### Q-Value Plots

The Q-value visualizations show a grid where each cell contains 5 colored triangles/circle:
- **Center circle**: NOOP action
- **Top triangle**: UP action
- **Bottom triangle**: DOWN action
- **Left triangle**: LEFT action
- **Right triangle**: RIGHT action

Colors indicate Q-values (blue = low, red = high).

### Policy Plots

The policy plots show:
- **Arrows**: Action probabilities (larger arrow = higher probability)
- **Green 'S'**: Start location
- **Red 'R'**: Reward location
- **Black squares**: Walls
- **Orange**: Lava

### Learning Curves

- **Returns plot**: Should increase over iterations
- **Normalized returns**: Should approach 1.0 (expert level)
- **Loss plot**: Should decrease as Q-values stabilize
- **Bellman error**: Measures how far from optimal Q* (should decrease)

## Advanced Usage

### Using the FQI Object

The script returns the FQI object for further analysis:

```python
fqi, log_dict, gridspec = run_fqi_on_maze(...)

# Access learned Q-values
learned_q = fqi.current_q  # Shape: (num_states, num_actions)

# Access optimal Q-values
optimal_q = fqi.ground_truth_q

# Evaluate the policy
returns = fqi.eval_policy(render=False, n_rollouts=100)

# Access the environment
env = fqi.env
```

### Different Network Types

```python
from debugq.models import q_networks

# Linear network (for wrapped observations)
network = q_networks.LinearNetwork(env)

# Fully connected network
network = q_networks.FCNetwork(env, layers=[64, 64])
```

### Weighted FQI

```python
from debugq.algos import exact_fqi

# Use weighted exact FQI with different weighting schemes
fqi = exact_fqi.WeightedExactFQI(
    env, 
    network,
    weighting_scheme='pi*',  # Options: 'uniform', 'pi*', 'pi', 'robust_adversarial'
    **fqi_args
)
```

### Custom Environments

```python
from rlutil.envs.gridcraft import grid_spec_cy, grid_env_cy
from rlutil.envs.gridcraft.grid_spec_cy import TileType

# Create a custom grid spec
gs = grid_spec_cy.spec_from_sparse_locations(
    10, 10,  # 10x10 grid
    {
        TileType.START: [(5, 5)],
        TileType.REWARD: [(0, 0), (9, 9)],
        TileType.WALL: [(i, 5) for i in range(10) if i != 5],
    }
)

env = grid_env_cy.GridEnv(gs)
```

### Example: Testing Different Entropy Weights

```python
for ent_wt in [0.0, 0.1, 0.5, 1.0]:
    maze_name = f"maze_ent_{ent_wt}"
    print(f"\nTraining with entropy weight = {ent_wt}")
    fqi, log_dict, gridspec = run_fqi_on_maze(
        maze_string=maze_string,
        maze_name=maze_name,
        ent_wt=ent_wt,
        num_iterations=30
    )
    print(f"Final return: {log_dict['returns'][-1]:.2f}")
```

## Repository Integration

The script demonstrates how to use several key components from the repository:

**Environments:**
- `rlutil.envs.gridcraft.grid_spec_cy`
- `rlutil.envs.gridcraft.grid_env_cy`
- `debugq.envs.time_limit_wrapper`

**Algorithms:**
- `debugq.algos.exact_fqi`
- `debugq.algos.stopping`

**Models:**
- `debugq.models.q_networks`

**Utilities:**
- `debugq.pytorch_util`
- `rlutil.logging.logger`
- `rlutil.logging.qval_plotter`
- `rlutil.envs.tabular.q_iteration`

## Design Decisions

### 1. Importing vs Copying
- **Decision**: Import from repository modules instead of copying code
- **Rationale**: 
  - Maintains code reusability
  - Easier to update
  - Follows DRY principle
  - Shows how to use the repository's API

### 2. Organized Output Structure
- **Decision**: Save outputs in maze-specific folders
- **Rationale**:
  - Easy to manage multiple experiments
  - Clear organization
  - No file name conflicts
  - Scalable for batch processing

### 3. PDF with Tight Bounding Boxes
- **Decision**: Save visualizations as PDFs instead of PNGs
- **Rationale**:
  - Publication-ready quality
  - Vector graphics (scalable)
  - Smaller file sizes
  - Professional appearance

### 4. Logger Integration
- **Decision**: Monkey-patch logger to capture metrics
- **Rationale**:
  - Non-invasive to existing code
  - Captures all metrics logged by FQI
  - Easy to extend
  - Maintains compatibility

## Troubleshooting

### Slow Training

- Reduce `max_project_steps`
- Reduce `num_iterations`
- Use a smaller maze

### Poor Performance

- Increase `num_iterations`
- Increase `max_project_steps` 
- Adjust `lr` (try 1e-3 to 1e-2)
- Check if maze is solvable

### Visualization Issues

- The script uses a non-interactive backend (`Agg`), so no windows will open
- All outputs are saved to the specified folder
- Check that the output directory has write permissions

### Memory Issues

- The script closes figures after saving to free memory
- For very large mazes or many experiments, consider processing in batches

## Code Structure

The script is organized into several modular functions:

- `create_maze_env()`: Creates the maze environment
- `visualize_qvalues()`: Plots Q-values on the grid
- `visualize_policy()`: Visualizes the policy as arrows
- `plot_learning_curves()`: Creates learning progress plots
- `run_fqi_on_maze()`: Main training loop
- `main()`: Entry point with example configuration

## Testing

The script has been tested with:
- Python 3.9+
- PyTorch (CPU and CUDA)
- Matplotlib 3.7+
- NumPy 1.x

All Cython extensions must be compiled first with `make build`.

## Future Enhancements

Possible extensions (not implemented):
1. Multi-environment comparison in a single plot
2. Hyperparameter sweep visualization
3. Real-time training visualization
4. Policy evaluation with rendering/video
5. Comparison of different network architectures
6. Replay buffer visualization
7. State visitation heatmaps
8. Trajectory visualization

## Further Reading

- **FQI Algorithm**: See `debugq/algos/fqi.py` and `debugq/algos/exact_fqi.py`
- **Grid Environments**: See `rlutil/envs/gridcraft/`
- **Q-Iteration**: See `rlutil/envs/tabular/q_iteration.py`

## Citation

If you use this code in your research, please cite the original repository.

## License

See LICENSE file in the repository root.
