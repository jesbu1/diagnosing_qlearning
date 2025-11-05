"""
Example script demonstrating history-conditioned FQI with trajectory jitter.

This script shows how to:
1. Load an optimal Q* from saved results
2. Create a history-conditioned Q-network
3. Train using FQI with jittered trajectories
"""

import numpy as np
from debugq.models.q_networks import HistoryConditionedQNetwork
from debugq.algos.history_fqi import HistoryConditionedFQI
from debugq.algos.replay_buffer_fqi import TrajectoryReplayBuffer
from rlutil.envs.gridcraft import grid_env, mazes
from rlutil.logging import logger


def main():
    # =====================================================================
    # 1. Setup Environment
    # =====================================================================
    print("Setting up environment...")
    env = grid_env.GridEnv(mazes.obstacle_maze())
    print(f"Environment: {env.num_states} states, {env.num_actions} actions")
    
    # =====================================================================
    # 2. Create History-Conditioned Q-Network
    # =====================================================================
    print("\nCreating history-conditioned Q-network...")
    history_length = 10
    network = HistoryConditionedQNetwork(
        env,
        history_encoder='lstm',  # or 'gru'
        hidden_dim=64,
        num_layers=1
    )
    print(f"Network created with history_length={history_length}")
    
    # =====================================================================
    # 3. Setup Replay Buffer
    # =====================================================================
    print("\nSetting up trajectory replay buffer...")
    replay_buffer = TrajectoryReplayBuffer(capacity=100000)
    
    # =====================================================================
    # 4. Load Optimal Q* (if available)
    # =====================================================================
    # Path to your saved optimal Q-function
    optimal_q_path = "fqi_results/maze_with_obstacles/optimal_q.npy"
    
    # Check if file exists
    import os
    if not os.path.exists(optimal_q_path):
        print(f"\nWarning: Optimal Q* not found at {optimal_q_path}")
        print("Will use standard FQI targets instead.")
        optimal_q_path = None
        use_optimal_q_targets = False
    else:
        print(f"\nLoading optimal Q* from {optimal_q_path}")
        use_optimal_q_targets = True
    
    # =====================================================================
    # 5. Create History-Conditioned FQI Algorithm
    # =====================================================================
    print("\nCreating History-Conditioned FQI algorithm...")
    
    fqi_algo = HistoryConditionedFQI(
        env=env,
        network=network,
        history_length=history_length,
        replay_buffer=replay_buffer,
        batch_size=32,
        
        # Jitter augmentation parameters
        jitter_prob=0.5,  # Apply jitter to 50% of samples
        jitter_type='mixed',  # 'rewind', 'pause', 'mixed', or 'none'
        jitter_kwargs={
            'rewind_steps': 3,      # For rewind jitter
            'pause_duration': 2,    # For pause jitter
            'num_pauses': 1,        # Number of pause points
            'rewind_prob': 1.0,
            'pause_prob': 1.0
        },
        
        # Q* loading parameters
        optimal_q_path=optimal_q_path,
        use_optimal_q_targets=use_optimal_q_targets,
        
        # Training parameters
        collect_trajectories_per_iter=10,  # Collect 10 new trajectories per iteration
        lr=1e-3,
        discount=0.99,
        ent_wt=1.0,
        min_project_steps=100,
        max_project_steps=1000,
        n_steps=1,
        
        # FQI-specific
        backup_mode='exact',  # or 'sampling'
    )
    
    print("\nFQI algorithm created with parameters:")
    print(f"  - History length: {history_length}")
    print(f"  - Batch size: 32")
    print(f"  - Jitter probability: 0.5")
    print(f"  - Jitter type: mixed")
    print(f"  - Using optimal Q* targets: {use_optimal_q_targets}")
    
    # =====================================================================
    # 6. Train the Algorithm
    # =====================================================================
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    num_iterations = 50
    
    logger.configure(dir='./logs/history_fqi_example', format_strs=['stdout', 'csv'])
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # Run one FQI update
        fqi_algo.update(step=iteration)
        
        # Log progress
        if iteration % 10 == 0:
            print(f"Current Q-value mean: {np.mean(fqi_algo.current_q):.4f}")
            print(f"Replay buffer size: {len(replay_buffer)}")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    
    # =====================================================================
    # 7. Evaluate Final Policy
    # =====================================================================
    print("\nEvaluating final policy...")
    final_returns = fqi_algo.eval_policy(render=False, n_rollouts=100)
    print(f"Final average return: {final_returns:.4f}")
    print(f"Expert return: {fqi_algo.expert_returns:.4f}")
    print(f"Random return: {fqi_algo.random_returns:.4f}")
    print(f"Normalized return: {fqi_algo.normalize_returns(final_returns):.4f}")
    
    # =====================================================================
    # 8. Example: Custom Update Rule for Jittered Trajectories
    # =====================================================================
    print("\n" + "="*70)
    print("CUSTOMIZATION NOTES:")
    print("="*70)
    print("""
To implement your custom update rule for jittered trajectories, you can:

1. Subclass HistoryConditionedFQI and override compute_weights():
   
   class CustomHistoryFQI(HistoryConditionedFQI):
       def compute_weights(self, history_states, history_actions, itr=0):
           # Your custom weighting logic here
           # You have access to:
           # - history_states: [batch, history_length]
           # - history_actions: [batch, history_length]
           # - self.ground_truth_timesteps (if stored)
           # - self.masks (if stored)
           
           weights = ... # Your computation
           return weights

2. Override evaluate_target() to customize how targets are computed:
   
   def evaluate_target(self, sample_histories, sample_actions, 
                      sample_next_states, sample_rewards,
                      masks=None, ground_truth_timesteps=None, mode=None):
       # Use masks and ground_truth_timesteps to handle jittered samples
       # differently from original samples
       
       # Example: Only use non-jittered samples for targets
       is_jittered = (masks.sum(axis=1) < self.history_length)
       
       # Your custom target computation
       target_q = ...
       return target_q

3. The masks and ground_truth_timesteps are available in get_sample_states():
   - masks[i, j] = 0.0 if timestep j is jittered (rewound/paused)
   - masks[i, j] = 1.0 if timestep j is original
   - ground_truth_timesteps[i, j] = actual episode timestep
   
   Example jittered trajectory:
   Original: [s1, s2, s3, s4, s5]
   Rewound:  [s1, s2, s3, s4, s5, s4, s3, s2]
   
   masks:     [1,  1,  1,  1,  1,  0,  0,  0]
   gt_times:  [0,  1,  2,  3,  4,  3,  2,  1]
""")


if __name__ == "__main__":
    main()


