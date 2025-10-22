#!/usr/bin/env python
"""
Run Fitted Q-Iteration on a maze/grid environment with tabular model and visualize the results.

This script demonstrates:
1. Creating a maze environment from a string
2. Running exact FQI with a tabular network
3. Visualizing Q-values on the grid
4. Plotting learning curves
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as MPLGridSpec

# Import environment and algorithm components
from rlutil.envs.gridcraft import grid_spec_cy, grid_env_cy
from rlutil.envs.gridcraft.grid_spec_cy import TileType
from debugq.envs import time_limit_wrapper
from debugq.algos import exact_fqi, stopping
from debugq.models import q_networks
from debugq import pytorch_util as ptu
from rlutil.logging import logger, log_utils
from rlutil.logging.qval_plotter import TabularQValuePlotter
from rlutil.envs.tabular import q_iteration as q_iteration_py


def create_maze_env(maze_string, time_limit=50):
    """
    Create a tabular maze environment from a string specification.
    
    Args:
        maze_string: String defining the maze layout
                    'S' = start, 'R' = reward, 'O' = open space
                    '#' = wall, 'L' = lava
        time_limit: Maximum steps per episode
        
    Returns:
        Wrapped environment ready for FQI
    """
    # Create grid specification from string
    gridspec = grid_spec_cy.spec_from_string(maze_string)
    
    # Create the tabular grid environment
    env = grid_env_cy.GridEnv(gridspec)
    
    # Wrap with time limit
    env = time_limit_wrapper.TimeLimitWrapper(env, time_limit=time_limit)
    
    return env, gridspec


def visualize_qvalues(q_values, gridspec, title="Q-Values", save_path=None):
    """
    Visualize Q-values on the grid using TabularQValuePlotter.
    
    Args:
        q_values: numpy array of shape (num_states, num_actions)
        gridspec: GridSpec object defining the grid
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    grid_width, grid_height = gridspec.data.shape
    plotter = TabularQValuePlotter(
        grid_width, 
        grid_height, 
        num_action=5, 
        invert_y=True,
        text_values=True
    )
    
    # Set Q-values for each state-action pair
    for state in range(len(gridspec)):
        xy = gridspec.idx_to_xy(state)
        x, y = xy  # idx_to_xy returns a tuple (x, y)
        for action in range(5):
            plotter.set_value(x, y, action, q_values[state, action])
    
    # Create the plot
    plotter.make_plot()
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    return plotter


def visualize_policy(q_values, gridspec, ent_wt=0.1, title="Policy", save_path=None):
    """
    Visualize the policy derived from Q-values.
    
    Args:
        q_values: numpy array of shape (num_states, num_actions)
        gridspec: GridSpec object defining the grid
        ent_wt: Entropy weight for soft policy
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    # Get policy probabilities
    policy = q_iteration_py.get_policy(q_values, ent_wt=ent_wt)
    
    # Plot policy as arrows on grid
    grid_width, grid_height = gridspec.data.shape
    fig, ax = plt.subplots(figsize=(grid_width, grid_height))
    
    # Action vectors for visualization
    action_vectors = {
        0: (0, 0),      # NOOP
        1: (0, -0.3),   # UP
        2: (0, 0.3),    # DOWN
        3: (-0.3, 0),   # LEFT
        4: (0.3, 0),    # RIGHT
    }
    
    for state in range(len(gridspec)):
        xy = gridspec.idx_to_xy(state)
        x, y = xy  # idx_to_xy returns a tuple (x, y)
        
        # Get tile type to determine if we should draw
        tile = gridspec.get_value((x, y))
        if tile == TileType.WALL:
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black'))
            continue
        
        # Draw arrows weighted by policy probabilities
        for action in range(5):
            prob = policy[state, action]
            if prob > 0.01:  # Only draw significant probabilities
                dx, dy = action_vectors[action]
                dx *= prob
                dy *= prob
                if action == 0:  # NOOP as circle
                    circle = plt.Circle((x, y), prob * 0.1, color='blue', alpha=0.5)
                    ax.add_patch(circle)
                else:
                    ax.arrow(x, y, dx, dy, head_width=0.1*prob, head_length=0.1*prob,
                            fc='blue', ec='blue', alpha=0.7)
        
        # Mark special tiles
        if tile == TileType.START:
            ax.text(x, y, 'S', ha='center', va='center', fontsize=12, color='green', weight='bold')
        elif tile == TileType.REWARD:
            ax.text(x, y, 'R', ha='center', va='center', fontsize=12, color='red', weight='bold')
        elif tile == TileType.LAVA:
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='orange', alpha=0.3))
    
    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    return fig, ax


def plot_learning_curves(log_dict, save_path=None):
    """
    Plot learning curves from FQI training.
    
    Args:
        log_dict: Dictionary of logged values
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = MPLGridSpec(3, 2, figure=fig)
    
    # Plot returns
    ax1 = fig.add_subplot(gs[0, 0])
    if 'returns' in log_dict:
        ax1.plot(log_dict['returns'], label='Returns', linewidth=2)
        ax1.axhline(log_dict['returns_expert'][0] if 'returns_expert' in log_dict else 0, 
                   color='g', linestyle='--', label='Expert')
        ax1.axhline(log_dict['returns_random'][0] if 'returns_random' in log_dict else 0,
                   color='r', linestyle='--', label='Random')
    ax1.set_xlabel('FQI Iteration')
    ax1.set_ylabel('Return')
    ax1.set_title('Learning Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot normalized returns
    ax2 = fig.add_subplot(gs[0, 1])
    if 'returns_normalized' in log_dict:
        ax2.plot(log_dict['returns_normalized'], linewidth=2, color='purple')
        ax2.axhline(1.0, color='g', linestyle='--', label='Expert level')
        ax2.axhline(0.0, color='r', linestyle='--', label='Random level')
    ax2.set_xlabel('FQI Iteration')
    ax2.set_ylabel('Normalized Return')
    ax2.set_title('Normalized Learning Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot projection loss
    ax3 = fig.add_subplot(gs[1, 0])
    if 'project_loss' in log_dict:
        ax3.semilogy(log_dict['project_loss'], linewidth=2, color='orange')
    ax3.set_xlabel('FQI Iteration')
    ax3.set_ylabel('Projection Loss (log scale)')
    ax3.set_title('TD Error / Projection Loss')
    ax3.grid(True, alpha=0.3)
    
    # Plot Q-value statistics
    ax4 = fig.add_subplot(gs[1, 1])
    if 'fit_q_value_mean' in log_dict:
        ax4.plot(log_dict['fit_q_value_mean'], label='Fitted Q mean', linewidth=2)
    if 'target_q_value_mean' in log_dict:
        ax4.plot(log_dict['target_q_value_mean'], label='Target Q mean', linewidth=2)
    ax4.set_xlabel('FQI Iteration')
    ax4.set_ylabel('Mean Q-Value')
    ax4.set_title('Q-Value Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot bellman error
    ax5 = fig.add_subplot(gs[2, 0])
    if 'tq_q*_diff_abs_mean' in log_dict:
        ax5.plot(log_dict['tq_q*_diff_abs_mean'], linewidth=2, color='red')
    ax5.set_xlabel('FQI Iteration')
    ax5.set_ylabel('|T*Q - Q*|')
    ax5.set_title('Bellman Error to Optimal Q*')
    ax5.grid(True, alpha=0.3)
    
    # Plot fit steps
    ax6 = fig.add_subplot(gs[2, 1])
    if 'fit_steps' in log_dict:
        ax6.plot(log_dict['fit_steps'], linewidth=2, color='brown')
    ax6.set_xlabel('FQI Iteration')
    ax6.set_ylabel('Gradient Steps')
    ax6.set_title('Projection Steps per Iteration')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    return fig


def run_fqi_on_maze(maze_string, 
                    num_iterations=50,
                    time_limit=50,
                    discount=0.99,
                    ent_wt=0.1,
                    lr=5e-3,
                    min_project_steps=10,
                    max_project_steps=100,
                    visualize_every=10,
                    maze_name="default_maze",
                    output_dir="fqi_results"):
    """
    Run FQI on a maze environment and visualize results.
    
    Args:
        maze_string: String defining the maze
        num_iterations: Number of FQI iterations
        time_limit: Episode time limit
        discount: Discount factor
        ent_wt: Entropy weight for soft Q-learning
        lr: Learning rate for projection
        min_project_steps: Minimum gradient steps per projection
        max_project_steps: Maximum gradient steps per projection
        visualize_every: Visualize Q-values every N iterations
        maze_name: Name for this maze (used for output folder)
        output_dir: Base directory for outputs
    """
    print("="*80)
    print("Running Fitted Q-Iteration on Maze Environment")
    print("="*80)
    
    # Create environment
    print("\n1. Creating maze environment...")
    env, gridspec = create_maze_env(maze_string, time_limit=time_limit)
    grid_width, grid_height = gridspec.data.shape
    print(f"   - Grid size: {grid_width} x {grid_height}")
    print(f"   - Num states: {env.num_states}")
    print(f"   - Num actions: {env.num_actions}")
    
    # Create network
    print("\n2. Creating tabular Q-network...")
    network = q_networks.TabularNetwork(env)
    ptu.initialize_network(network)
    print(f"   - Network parameters: {sum(p.numel() for p in network.parameters())}")
    
    # Setup FQI algorithm
    print("\n3. Setting up FQI algorithm...")
    fqi_args = {
        'min_project_steps': min_project_steps,
        'max_project_steps': max_project_steps,
        'lr': lr,
        'discount': discount,
        'n_steps': 1,
        'ent_wt': ent_wt,
        'stop_modes': (stopping.AtolStop(atol=1e-4), stopping.RtolStop(rtol=1e-3)),
        'backup_mode': 'exact',
    }
    
    fqi = exact_fqi.ExactFQI(env, network, **fqi_args)
    print(f"   - Discount: {discount}")
    print(f"   - Entropy weight: {ent_wt}")
    print(f"   - Learning rate: {lr}")
    print(f"   - Expert returns: {fqi.expert_returns:.3f}")
    print(f"   - Random returns: {fqi.random_returns:.3f}")
    
    # Train FQI
    print(f"\n4. Training FQI for {num_iterations} iterations...")
    log_dict = {
        'returns': [],
        'returns_normalized': [],
        'returns_expert': [],
        'returns_random': [],
        'project_loss': [],
        'fit_steps': [],
        'fit_q_value_mean': [],
        'target_q_value_mean': [],
        'tq_q*_diff_abs_mean': [],
    }
    
    # Monkey-patch dump_tabular to capture values before they're cleared
    original_dump_tabular = logger.dump_tabular
    captured_dict = {}
    def capturing_dump_tabular(*args, **kwargs):
        nonlocal captured_dict
        captured_dict = dict(logger._tabular)
        return original_dump_tabular(*args, **kwargs)
    logger.dump_tabular = capturing_dump_tabular
    
    for i in range(num_iterations):
        # Run one FQI update
        fqi.update(step=i)
        
        # Collect logged values that were captured before clearing
        log_dict['returns'].append(float(captured_dict.get('returns', 0)))
        log_dict['returns_normalized'].append(float(captured_dict.get('returns_normalized', 0)))
        log_dict['returns_expert'].append(float(captured_dict.get('returns_expert', 0)))
        log_dict['returns_random'].append(float(captured_dict.get('returns_random', 0)))
        log_dict['project_loss'].append(float(captured_dict.get('project_loss', 0)))
        log_dict['fit_steps'].append(int(captured_dict.get('fit_steps', 0)))
        log_dict['fit_q_value_mean'].append(float(captured_dict.get('fit_q_value_mean', 0)))
        log_dict['target_q_value_mean'].append(float(captured_dict.get('target_q_value_mean', 0)))
        log_dict['tq_q*_diff_abs_mean'].append(float(captured_dict.get('tq_q*_diff_abs_mean', 0)))
        
        # Print progress
        if (i+1) % 5 == 0 or i == 0:
            print(f"   Iter {i+1:3d}: Return={log_dict['returns'][-1]:7.3f} "
                  f"(norm={log_dict['returns_normalized'][-1]:5.3f}), "
                  f"Loss={log_dict['project_loss'][-1]:.4e}, "
                  f"Steps={log_dict['fit_steps'][-1]}")
    
    # Restore original dump_tabular
    logger.dump_tabular = original_dump_tabular
    
    print("\n5. Saving results...")
    
    # Create output directory for this maze
    maze_output_dir = os.path.join(output_dir, maze_name)
    os.makedirs(maze_output_dir, exist_ok=True)
    print(f"   - Output directory: {maze_output_dir}/")
    
    # Save Q-values
    print("   - Saving Q-values...")
    np.savez(
        os.path.join(maze_output_dir, 'q_values.npz'),
        learned_q=fqi.current_q,
        optimal_q=fqi.ground_truth_q,
        expert_returns=fqi.expert_returns,
        random_returns=fqi.random_returns,
        final_returns=log_dict['returns'][-1],
        final_normalized_returns=log_dict['returns_normalized'][-1],
    )
    
    # Also save as separate .npy files for easier loading
    np.save(os.path.join(maze_output_dir, 'learned_q.npy'), fqi.current_q)
    np.save(os.path.join(maze_output_dir, 'optimal_q.npy'), fqi.ground_truth_q)
    
    # Create visualizations
    plt.close('all')
    
    # Plot learning curves
    print("   - Generating learning curves...")
    plot_learning_curves(log_dict, save_path=os.path.join(maze_output_dir, 'learning_curves.pdf'))
    plt.close()
    
    # Visualize optimal Q-values
    print("   - Visualizing optimal Q-values...")
    plt.figure(figsize=(10, 8))
    visualize_qvalues(fqi.ground_truth_q, gridspec, 
                     title="Optimal Q-Values (Q*)",
                     save_path=os.path.join(maze_output_dir, 'qstar.pdf'))
    plt.close()
    
    # Visualize learned Q-values
    print("   - Visualizing learned Q-values...")
    plt.figure(figsize=(10, 8))
    visualize_qvalues(fqi.current_q, gridspec,
                     title=f"Learned Q-Values (Iteration {num_iterations})",
                     save_path=os.path.join(maze_output_dir, 'learned_q.pdf'))
    plt.close()
    
    # Visualize policy
    print("   - Visualizing learned policy...")
    plt.figure(figsize=(10, 8))
    visualize_policy(fqi.current_q, gridspec, ent_wt=ent_wt,
                    title="Learned Policy",
                    save_path=os.path.join(maze_output_dir, 'policy.pdf'))
    plt.close()
    
    # Visualize optimal policy
    print("   - Visualizing optimal policy...")
    plt.figure(figsize=(10, 8))
    visualize_policy(fqi.ground_truth_q, gridspec, ent_wt=ent_wt,
                    title="Optimal Policy",
                    save_path=os.path.join(maze_output_dir, 'optimal_policy.pdf'))
    plt.close()
    
    print("\n" + "="*80)
    print("Training complete! Results saved to:")
    print(f"\nQ-values:")
    print(f"  - {maze_output_dir}/q_values.npz: All Q-values and metrics")
    print(f"  - {maze_output_dir}/learned_q.npy: Learned Q-values")
    print(f"  - {maze_output_dir}/optimal_q.npy: Optimal Q-values")
    print(f"\nVisualizations:")
    print(f"  - {maze_output_dir}/learning_curves.pdf: Learning progress")
    print(f"  - {maze_output_dir}/qstar.pdf: Optimal Q-values")
    print(f"  - {maze_output_dir}/learned_q.pdf: Learned Q-values")
    print(f"  - {maze_output_dir}/policy.pdf: Learned policy")
    print(f"  - {maze_output_dir}/optimal_policy.pdf: Optimal policy")
    print("="*80)
    
    return fqi, log_dict, gridspec


def main():
    """Main function to run the FQI maze demo."""
    
    # Define a simple maze
    # You can modify this string to create different mazes
    # S = Start, R = Reward, O = Open space, # = Wall, L = Lava
    
    # Example 1: Simple 3x2 maze
    # maze_string = "SOO\\OOR\\"
    
    # Example 2: Larger 6x4 maze
    maze_string = "SOOOOO\\OOOOOO\\OOOOOO\\OOOOOR\\"
    
    # Example 3: Maze with obstacles
    # maze_string = "SOOO\\O##O\\O##O\\OOOR\\"
    
    # Example 4: Maze with lava
    # maze_string = "OOOOOOR\\SOLLLLL\\OOOOOOO\\OOOOOO3\\"
    
    print("\nMaze Layout:")
    print("-" * 40)
    for row in maze_string.rstrip('\\').split('\\'):
        print(row)
    print("-" * 40)
    
    # Run FQI with a name for this maze
    maze_name = "maze_6x4"  # Change this for different mazes
    
    fqi, log_dict, gridspec = run_fqi_on_maze(
        maze_string=maze_string,
        num_iterations=50,
        time_limit=50,
        discount=0.95,
        ent_wt=0.1,
        lr=5e-3,
        min_project_steps=10,
        max_project_steps=100,
        maze_name=maze_name,
        output_dir="fqi_results",
    )
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"  Expert returns: {fqi.expert_returns:.3f}")
    print(f"  Random returns: {fqi.random_returns:.3f}")
    print(f"  Final returns: {log_dict['returns'][-1]:.3f}")
    print(f"  Final normalized returns: {log_dict['returns_normalized'][-1]:.3f}")
    print(f"  Final Bellman error: {log_dict['tq_q*_diff_abs_mean'][-1]:.4e}")


if __name__ == "__main__":
    main()

