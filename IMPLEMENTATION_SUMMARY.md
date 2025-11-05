# Implementation Summary: History-Conditioned FQI with Trajectory Jitter

## What Was Changed

### New Files Created

1. **`debugq/algos/trajectory_augmentation.py`**
   - Functions for generating rewound and paused trajectories
   - `rewind_trajectory()` - Append reversed history
   - `pause_trajectory()` - Insert repeated states
   - `apply_jitter()` - Apply random jitter types
   - Returns: augmented states/actions, ground truth timesteps, and masks

2. **`debugq/algos/history_fqi.py`**
   - Main `HistoryConditionedFQI` class
   - Extends base `FQI` class
   - Handles trajectory collection, jitter augmentation, and history-based training
   - **Key methods to customize:**
     - `compute_weights()` - Custom sample weighting
     - `evaluate_target()` - Custom target computation
     - `get_sample_states()` - Modify sampling behavior

3. **`example_history_fqi.py`**
   - Complete working example
   - Shows how to set up and train the algorithm
   - Includes notes on customization

4. **`HISTORY_FQI_README.md`**
   - Comprehensive documentation
   - Usage examples
   - Customization guide

5. **`debugq/algos/test_history_fqi.py`**
   - Unit tests for all components
   - Run with: `pytest debugq/algos/test_history_fqi.py`

### Modified Files

1. **`debugq/models/q_networks.py`**
   - **Added:** `HistoryConditionedQNetwork` class
   - LSTM/GRU-based encoder for processing observation histories
   - Handles masking for valid timesteps
   - Forward pass: history_states [batch, history_length] → Q-values [batch, num_actions]

2. **`debugq/algos/replay_buffer_fqi.py`**
   - **Added:** `TrajectoryReplayBuffer` class
   - Stores complete episodes with temporal structure
   - `add_trajectory()` - Add full trajectories
   - `sample_history_segments()` - Sample with history context
   - Returns: history_states, history_actions, next_states, rewards, masks, timesteps

3. **`debugq/algos/utils.py`**
   - **Added:** `load_optimal_q(path)` - Load Q* from .npy/.npz files
   - **Added:** `collect_trajectories()` - Collect episodes by rolling out a policy

## Key Data Structures

### 1. History States
- **Shape:** `[batch_size, history_length]`
- **Type:** State indices (int)
- **Purpose:** Observation history for Q-function input

### 2. Masks
- **Shape:** `[batch_size, history_length]`
- **Type:** Binary (float32: 0.0 or 1.0)
- **Purpose:** Indicate which timesteps are real (1.0) vs jittered (0.0)
- **Example:**
  ```
  Original: [s1, s2, s3, s4, s5]
  Rewound:  [s1, s2, s3, s4, s5, s4, s3]
  Mask:     [1,  1,  1,  1,  1,  0,  0]
  ```

### 3. Ground Truth Timesteps
- **Shape:** `[batch_size, history_length]`
- **Type:** Timestep indices (int)
- **Purpose:** Map each position to its true episode timestep
- **Example:**
  ```
  Original: [s1, s2, s3, s4, s5]
  Rewound:  [s1, s2, s3, s4, s5, s4, s3]
  GT Times: [0,  1,  2,  3,  4,  3,  2]
  ```

## Where to Implement Your Custom Update Rule

### Option 1: Custom Weighting

Override `compute_weights()` in a subclass:

```python
class MyHistoryFQI(HistoryConditionedFQI):
    def compute_weights(self, history_states, history_actions, itr=0):
        """
        Assign custom weights to each sample.
        
        Access self.current_masks and self.current_gt_timesteps
        if you store them in get_sample_states().
        """
        batch_size = history_states.shape[0]
        weights = np.ones(batch_size)
        
        # Your custom logic here
        # Example: Weight based on amount of jitter
        for i in range(batch_size):
            num_real = self.current_masks[i].sum()
            num_jittered = self.history_length - num_real
            
            if num_jittered > 0:
                # Downweight heavily jittered samples
                weights[i] = 1.0 / (1.0 + num_jittered)
        
        return weights / weights.sum()  # Normalize
```

### Option 2: Custom Target Computation

Override `evaluate_target()` to handle jittered samples differently:

```python
class MyHistoryFQI(HistoryConditionedFQI):
    def evaluate_target(self, sample_histories, sample_actions,
                       sample_next_states, sample_rewards,
                       masks=None, ground_truth_timesteps=None, mode=None):
        """
        Compute targets differently for jittered vs clean samples.
        """
        batch_size = len(sample_next_states)
        targets = []
        
        for i in range(batch_size):
            # Check if sample is jittered
            if masks is not None:
                num_real = masks[i].sum()
                is_jittered = (num_real < self.history_length)
            else:
                is_jittered = False
            
            # Use different target for jittered samples
            if is_jittered:
                # Option A: Use current Q
                v_next = q_iteration.logsumexp(
                    self.current_q[sample_next_states[i]], 
                    alpha=self.ent_wt
                )
            else:
                # Option B: Use optimal Q*
                v_next = q_iteration.logsumexp(
                    self.optimal_q[sample_next_states[i]], 
                    alpha=self.ent_wt
                )
            
            target = sample_rewards[i] + self.discount * v_next
            targets.append(target)
        
        target_tensor = ptu.tensor(targets)
        
        if mode == fqi.MULTIPLE_HEADS:
            # Expand to all actions
            target_tensor = target_tensor.unsqueeze(1).expand(-1, self.env.num_actions)
        
        return target_tensor
```

### Option 3: Custom Loss Function

Override the `project()` method to implement a completely custom loss:

```python
class MyHistoryFQI(HistoryConditionedFQI):
    def project(self, network=None, optimizer=None, sampler=None):
        """
        Custom projection with your own loss function.
        """
        if network is None:
            network = self.network
        if optimizer is None:
            optimizer = self.qnet_optimizer
        if sampler is None:
            sampler = self.get_sample_states
        
        for k in range(self.max_project_steps):
            # Sample
            (sample_histories, sample_actions, sample_next_states, 
             sample_rewards, weights, masks, ground_truth_timesteps) = sampler(itr=k)
            
            # Evaluate Q-values
            q_values = self.evaluate_qvalues(
                sample_histories, sample_actions, masks=masks, network=network
            )
            
            # Compute targets
            target_q = self.evaluate_target(
                sample_histories, sample_actions, sample_next_states, sample_rewards,
                masks=masks, ground_truth_timesteps=ground_truth_timesteps
            ).detach()
            
            # YOUR CUSTOM LOSS HERE
            # Example: Different loss for jittered vs clean samples
            clean_mask = ptu.tensor(masks.sum(axis=1) == self.history_length, dtype=torch.float32)
            jittered_mask = 1.0 - clean_mask
            
            loss_clean = torch.mean((q_values - target_q) ** 2, dim=1)
            loss_jittered = torch.mean((q_values - target_q) ** 2, dim=1)
            
            critic_loss = torch.mean(
                clean_mask * loss_clean + 
                0.5 * jittered_mask * loss_jittered  # Half weight for jittered
            )
            
            # Backprop
            network.zero_grad()
            critic_loss.backward()
            optimizer.step()
        
        return None, critic_loss, k
```

### Option 4: Store Masks for Later Use

Modify `get_sample_states()` to store masks and timesteps:

```python
class MyHistoryFQI(HistoryConditionedFQI):
    def get_sample_states(self, itr=0):
        # Get samples from parent
        result = super().get_sample_states(itr=itr)
        
        (history_states, history_actions, next_states, rewards,
         weights, masks, ground_truth_timesteps) = result
        
        # Store for use in compute_weights() and evaluate_target()
        self.current_masks = masks
        self.current_gt_timesteps = ground_truth_timesteps
        
        # Optionally filter or modify samples here
        
        return result
```

## Loading Optimal Q*

Your saved Q* files are in `fqi_results/`:

```python
from debugq.algos.utils import load_optimal_q

# Load Q* for a specific maze
optimal_q = load_optimal_q('fqi_results/maze_with_lava/optimal_q.npy')

# Use in HistoryConditionedFQI
fqi = HistoryConditionedFQI(
    env=env,
    network=network,
    optimal_q_path='fqi_results/maze_with_lava/optimal_q.npy',
    use_optimal_q_targets=True,  # Use Q* for all targets
    # ... other params
)
```

Or load manually and use in custom logic:

```python
fqi = HistoryConditionedFQI(
    env=env,
    network=network,
    optimal_q_path='fqi_results/maze_with_lava/optimal_q.npy',
    use_optimal_q_targets=False,  # We'll use it manually
)

# Access in custom methods
class MyHistoryFQI(HistoryConditionedFQI):
    def evaluate_target(self, ...):
        # self.optimal_q is available here
        v_star = q_iteration.logsumexp(self.optimal_q[next_state], alpha=self.ent_wt)
        return reward + self.discount * v_star
```

## Running the Code

### Basic Usage

```python
from debugq.models.q_networks import HistoryConditionedQNetwork
from debugq.algos.history_fqi import HistoryConditionedFQI
from rlutil.envs.gridcraft import grid_env, mazes

# Setup
env = grid_env.GridEnv(mazes.obstacle_maze())
network = HistoryConditionedQNetwork(env, history_encoder='lstm', hidden_dim=64)

# Create FQI
fqi = HistoryConditionedFQI(
    env=env,
    network=network,
    history_length=10,
    jitter_prob=0.5,
    jitter_type='mixed',
    optimal_q_path='fqi_results/maze_with_obstacles/optimal_q.npy',
    use_optimal_q_targets=True
)

# Train
for i in range(100):
    fqi.update(step=i)
```

### With Custom Logic

```python
class MyCustomFQI(HistoryConditionedFQI):
    def compute_weights(self, history_states, history_actions, itr=0):
        # Your custom weighting
        return custom_weights
    
    def evaluate_target(self, ...):
        # Your custom target computation
        return custom_targets

# Use custom class
fqi = MyCustomFQI(
    env=env,
    network=network,
    history_length=10,
    # ... params
)
```

### Run Example

```bash
python example_history_fqi.py
```

### Run Tests

```bash
pytest debugq/algos/test_history_fqi.py -v
```

## Quick Reference: Method Override Points

| Method | Purpose | What to Return |
|--------|---------|----------------|
| `compute_weights()` | Sample weighting | `np.array([batch_size])` |
| `evaluate_target()` | Target Q-values | `torch.Tensor([batch_size, num_actions])` or `[batch_size]` |
| `get_sample_states()` | Modify sampling | Tuple of 7 arrays (histories, actions, next_states, rewards, weights, masks, timesteps) |
| `project()` | Custom loss/training loop | (stopped_mode, loss, num_steps) |
| `pre_project()` | Before each iteration | None |
| `post_project()` | After each iteration | None |

## Next Steps

1. **Review the example:** `example_history_fqi.py`
2. **Read the docs:** `HISTORY_FQI_README.md`
3. **Run the tests:** `pytest debugq/algos/test_history_fqi.py`
4. **Implement your custom logic** by subclassing `HistoryConditionedFQI`
5. **Experiment with jitter parameters** to see what works best

## Questions/Issues

The implementation is designed to be flexible. The key extension points are:

1. **`compute_weights()`** - Control how samples are weighted
2. **`evaluate_target()`** - Control target computation (use Q*, current Q, or custom)
3. **`get_sample_states()`** - Control what data is sampled and how
4. **`project()`** - Full control over training loop

All of these have access to:
- `masks` - Which timesteps are real vs jittered
- `ground_truth_timesteps` - True temporal location
- `self.optimal_q` - Loaded optimal Q-function
- `self.current_q` - Current learned Q-function

You have complete flexibility to implement any update rule you want!


