# History-Conditioned FQI with Trajectory Jitter

This document describes the implementation of history-conditioned Fitted Q-Iteration (FQI) with support for trajectory jitter (rewinding and pausing).

## Overview

The implementation allows training Q-functions that condition on arbitrary-length observation histories, and supports generating trajectories with temporal perturbations (jitter) for robust learning.

## New Components

### 1. `HistoryConditionedQNetwork` (`debugq/models/q_networks.py`)

A Q-network that takes sequences of observations as input and outputs Q-values.

**Features:**
- LSTM or GRU encoder for processing history
- Handles variable-length histories with masking
- Converts state indices to observations automatically

**Usage:**
```python
from debugq.models.q_networks import HistoryConditionedQNetwork

network = HistoryConditionedQNetwork(
    env,
    history_encoder='lstm',  # or 'gru'
    hidden_dim=64,
    num_layers=1
)

# Forward pass
q_values = network(history_states, mask=mask)
# Input: history_states [batch, history_length]
# Optional: mask [batch, history_length] indicating valid timesteps
# Output: q_values [batch, num_actions]
```

### 2. `TrajectoryReplayBuffer` (`debugq/algos/replay_buffer_fqi.py`)

A replay buffer that stores complete trajectories and samples history segments.

**Features:**
- Stores full episodes with temporal structure
- Samples trajectory segments with sufficient history
- Returns masks and ground truth timesteps
- Automatic padding for short trajectories

**Usage:**
```python
from debugq.algos.replay_buffer_fqi import TrajectoryReplayBuffer

buffer = TrajectoryReplayBuffer(capacity=100000)

# Add trajectories
buffer.add_trajectory(states, actions, next_states, rewards, dones)

# Sample history segments
(history_states, history_actions, next_states, rewards, 
 masks, timesteps) = buffer.sample_history_segments(
    batch_size=32,
    history_length=10
)
```

### 3. Trajectory Augmentation (`debugq/algos/trajectory_augmentation.py`)

Functions for creating jittered trajectories.

**Jitter Types:**

**Rewind:** Append reversed history to the end
```python
# Original: [s1, s2, s3, s4, s5]
# Rewound:  [s1, s2, s3, s4, s5, s4, s3, s2]
# Masks:    [1,  1,  1,  1,  1,  0,  0,  0]
# GT times: [0,  1,  2,  3,  4,  3,  2,  1]

states, actions, gt_times, mask = rewind_trajectory(
    states, actions, 
    rewind_steps=3,
    rewind_prob=1.0
)
```

**Pause:** Repeat states at certain positions
```python
# Original: [s1, s2, s3, s4, s5]
# Paused:   [s1, s2, s3, s3, s3, s4, s5]
# Masks:    [1,  1,  1,  0,  0,  1,  1]
# GT times: [0,  1,  2,  2,  2,  3,  4]

states, actions, gt_times, mask = pause_trajectory(
    states, actions,
    pause_duration=3,
    num_pauses=1,
    pause_prob=1.0
)
```

**Mixed:** Random choice between rewind and pause
```python
states, actions, gt_times, mask = apply_jitter(
    states, actions,
    jitter_type='mixed'  # or 'rewind', 'pause', 'none'
)
```

### 4. `HistoryConditionedFQI` (`debugq/algos/history_fqi.py`)

Main algorithm class that ties everything together.

**Key Features:**
- Collects trajectories using current policy
- Stores in trajectory replay buffer
- Samples with history context
- Applies jitter augmentation
- Supports loading optimal Q* for targets
- Customizable weighting schemes

**Usage:**
```python
from debugq.algos.history_fqi import HistoryConditionedFQI

fqi = HistoryConditionedFQI(
    env=env,
    network=network,
    history_length=10,
    batch_size=32,
    
    # Jitter parameters
    jitter_prob=0.5,
    jitter_type='mixed',
    jitter_kwargs={
        'rewind_steps': 3,
        'pause_duration': 2
    },
    
    # Load optimal Q* for targets
    optimal_q_path='fqi_results/maze_with_lava/optimal_q.npy',
    use_optimal_q_targets=True,
    
    # Standard FQI params
    lr=1e-3,
    discount=0.99,
    ent_wt=1.0,
    max_project_steps=1000
)

# Train
for i in range(num_iterations):
    fqi.update(step=i)
```

### 5. Utility Functions (`debugq/algos/utils.py`)

Added helper functions:

**Load Optimal Q:**
```python
from debugq.algos.utils import load_optimal_q

optimal_q = load_optimal_q('fqi_results/maze_with_lava/optimal_q.npy')
# Returns: numpy array [num_states, num_actions]
# Supports .npy and .npz formats
```

**Collect Trajectories:**
```python
from debugq.algos.utils import collect_trajectories

trajectories = collect_trajectories(
    env, q_fn, 
    num_trajectories=10, 
    ent_wt=1.0
)
# Returns: List of (states, actions, next_states, rewards, dones) tuples
```

## Data Flow

### Standard FQI
```
Sample transitions → Train Q-network → Evaluate
        ↓
    (s, a, s', r)
```

### History-Conditioned FQI with Jitter
```
Collect trajectories → Store in buffer → Sample history segments
                                              ↓
                                      Apply jitter (optional)
                                              ↓
                          [history_states, masks, gt_timesteps]
                                              ↓
                            History Q-network(history) → Q-values
                                              ↓
                            Compute targets using Q* or current Q
                                              ↓
                                    Train with custom weights
```

## Customization: Implementing Your Update Rule

You have full control over how jittered trajectories are treated. Here are the key extension points:

### 1. Custom Sample Weighting

Override `compute_weights()` to weight jittered vs. non-jittered samples:

```python
class MyHistoryFQI(HistoryConditionedFQI):
    def compute_weights(self, history_states, history_actions, itr=0):
        """
        Compute weights for each sample.
        
        Args:
            history_states: [batch, history_length]
            history_actions: [batch, history_length]
        
        Access stored data:
            - self.current_masks: [batch, history_length]
            - self.current_gt_timesteps: [batch, history_length]
        """
        batch_size = history_states.shape[0]
        weights = np.ones(batch_size)
        
        # Example: Downweight jittered samples
        for i in range(batch_size):
            num_jittered = (self.current_masks[i] == 0).sum()
            if num_jittered > 0:
                weights[i] = 0.5  # Half weight for jittered
        
        return weights / weights.sum()  # Normalize
```

### 2. Custom Target Computation

Override `evaluate_target()` to use masks and ground truth timesteps:

```python
class MyHistoryFQI(HistoryConditionedFQI):
    def evaluate_target(self, sample_histories, sample_actions,
                       sample_next_states, sample_rewards,
                       masks=None, ground_truth_timesteps=None, mode=None):
        """
        Custom target computation.
        
        Args:
            sample_histories: [batch, history_length]
            sample_actions: [batch, history_length]
            sample_next_states: [batch]
            sample_rewards: [batch]
            masks: [batch, history_length] - 1=real, 0=jittered
            ground_truth_timesteps: [batch, history_length]
        """
        # Example: Use optimal Q* only for non-jittered samples
        batch_size = sample_histories.shape[0]
        targets = []
        
        for i in range(batch_size):
            is_jittered = (masks[i].sum() < self.history_length)
            
            if is_jittered:
                # Use current Q for jittered samples
                v_next = q_iteration.logsumexp(
                    self.current_q[sample_next_states[i]], 
                    alpha=self.ent_wt
                )
            else:
                # Use optimal Q* for clean samples
                v_next = q_iteration.logsumexp(
                    self.optimal_q[sample_next_states[i]], 
                    alpha=self.ent_wt
                )
            
            target = sample_rewards[i] + self.discount * v_next
            targets.append(target)
        
        return ptu.tensor(targets)
```

### 3. Store Masks/Timesteps in get_sample_states()

Modify `get_sample_states()` to store masks for use in other methods:

```python
class MyHistoryFQI(HistoryConditionedFQI):
    def get_sample_states(self, itr=0):
        # Call parent to get samples
        result = super().get_sample_states(itr=itr)
        
        (history_states, history_actions, next_states, rewards,
         weights, masks, ground_truth_timesteps) = result
        
        # Store for use in other methods
        self.current_masks = masks
        self.current_gt_timesteps = ground_truth_timesteps
        
        # You can modify any of these before returning
        # Example: Filter out heavily jittered samples
        valid_indices = []
        for i in range(len(masks)):
            if masks[i].sum() / self.history_length > 0.5:
                valid_indices.append(i)
        
        if len(valid_indices) > 0:
            history_states = history_states[valid_indices]
            history_actions = history_actions[valid_indices]
            next_states = next_states[valid_indices]
            rewards = rewards[valid_indices]
            weights = weights[valid_indices]
            masks = masks[valid_indices]
            ground_truth_timesteps = ground_truth_timesteps[valid_indices]
        
        return (history_states, history_actions, next_states, rewards,
                weights, masks, ground_truth_timesteps)
```

## Understanding Masks and Ground Truth Timesteps

### Masks
- Shape: `[batch_size, history_length]`
- Values: `1.0` for original timesteps, `0.0` for jittered timesteps
- Use case: Identify which parts of the history are real vs. synthetic

### Ground Truth Timesteps
- Shape: `[batch_size, history_length]`
- Values: Integer timestep indices from the original episode
- Use case: Map jittered positions back to their true temporal location

### Example

Original trajectory: `[s0, s1, s2, s3, s4]`

**After rewinding with rewind_steps=2:**
```
Augmented:  [s0, s1, s2, s3, s4, s3, s2]
Masks:      [1,  1,  1,  1,  1,  0,  0]
GT Times:   [0,  1,  2,  3,  4,  3,  2]
```

The last two positions are rewound (mask=0), and their ground truth timesteps show they correspond to t=3 and t=2.

**After pausing at position 2 with pause_duration=2:**
```
Augmented:  [s0, s1, s2, s2, s2, s3, s4]
Masks:      [1,  1,  1,  0,  0,  1,  1]
GT Times:   [0,  1,  2,  2,  2,  3,  4]
```

Positions 3 and 4 are paused repetitions (mask=0) of state s2 (gt_time=2).

## Loading Optimal Q*

The optimal Q-functions are stored in `fqi_results/`:

```
fqi_results/
├── maze_with_lava/
│   └── optimal_q.npy          # Shape: [num_states, num_actions]
├── maze_with_obstacles/
│   └── optimal_q.npy
└── larger_maze/
    └── optimal_q.npy
```

Load and use:
```python
from debugq.algos.utils import load_optimal_q

optimal_q = load_optimal_q('fqi_results/maze_with_lava/optimal_q.npy')

# Use in target computation
v_next = q_iteration.logsumexp(optimal_q[next_state], alpha=ent_wt)
target = reward + discount * v_next
```

## Example Script

See `example_history_fqi.py` for a complete working example.

Run it:
```bash
python example_history_fqi.py
```

## Testing

Unit tests for key components:

```bash
# Test trajectory augmentation
python -m pytest debugq/algos/test_trajectory_augmentation.py

# Test replay buffer
python -m pytest debugq/algos/test_replay_buffer.py

# Test network
python -m pytest debugq/models/test_history_network.py
```

## Architecture Details

### HistoryConditionedQNetwork

```
Input: state_indices [batch, history_length]
   ↓
Lookup: observations [batch, history_length, obs_dim]
   ↓
LSTM/GRU Encoder: [batch, history_length, hidden_dim]
   ↓
Select last valid timestep (using mask)
   ↓
MLP Head: [batch, hidden_dim] → [batch, num_actions]
   ↓
Output: Q-values [batch, num_actions]
```

### Training Loop

```python
for iteration in range(num_iterations):
    # 1. Collect new trajectories
    trajectories = collect_trajectories(env, current_q, num_trajectories=10)
    
    # 2. Add to replay buffer
    for traj in trajectories:
        replay_buffer.add_trajectory(*traj)
    
    # 3. Sample history segments with jitter
    samples = replay_buffer.sample_history_segments(batch_size, history_length)
    if jitter_prob > 0:
        samples = apply_jitter_augmentation(samples)
    
    # 4. Train Q-network
    for k in range(num_gradient_steps):
        q_values = network(history_states, masks)
        targets = compute_targets(next_states, rewards)
        loss = weighted_mse(q_values, targets, weights)
        loss.backward()
        optimizer.step()
    
    # 5. Update current Q for next iteration
    current_q = extract_q_values(network)
```

## Key Differences from Standard FQI

| Aspect | Standard FQI | History-Conditioned FQI |
|--------|--------------|-------------------------|
| Input | Single state `s` | History `[s_{t-k}, ..., s_t]` |
| Network | MLP: state → Q-values | LSTM/GRU: history → Q-values |
| Buffer | Transitions `(s, a, s', r)` | Trajectories with temporal structure |
| Sampling | Random transitions | History segments with context |
| Augmentation | None | Rewind/pause jitter |
| Targets | Q*(s') or backup | Q*(s') using ground truth state |

## Next Steps

1. **Implement your custom update rule** by subclassing `HistoryConditionedFQI`
2. **Tune jitter parameters** (rewind_steps, pause_duration, jitter_prob)
3. **Experiment with different history lengths**
4. **Try different network architectures** (LSTM vs. GRU, hidden_dim, num_layers)
5. **Customize weighting schemes** for jittered vs. clean samples

## Questions?

The code is well-documented with docstrings. Key files to read:
- `debugq/algos/history_fqi.py` - Main algorithm
- `debugq/algos/trajectory_augmentation.py` - Jitter functions
- `debugq/models/q_networks.py` - Network architecture
- `example_history_fqi.py` - Complete working example


