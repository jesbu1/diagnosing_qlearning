# Quick Start: History-Conditioned FQI

## Overview

This implementation allows Q-functions to condition on observation histories with support for trajectory jitter (rewinding/pausing) to simulate temporal perturbations.

## Files Created

```
debugq/
├── models/
│   └── q_networks.py                    [MODIFIED] Added HistoryConditionedQNetwork
└── algos/
    ├── trajectory_augmentation.py       [NEW] Rewind/pause jitter functions
    ├── history_fqi.py                   [NEW] Main algorithm
    ├── test_history_fqi.py              [NEW] Unit tests
    ├── utils.py                         [MODIFIED] Added load_optimal_q()
    └── replay_buffer_fqi.py             [MODIFIED] Added TrajectoryReplayBuffer

example_history_fqi.py                   [NEW] Complete working example
HISTORY_FQI_README.md                    [NEW] Full documentation
IMPLEMENTATION_SUMMARY.md                [NEW] What changed and where
QUICKSTART.md                            [NEW] This file
```

## Minimal Example

```python
from debugq.models.q_networks import HistoryConditionedQNetwork
from debugq.algos.history_fqi import HistoryConditionedFQI
from rlutil.envs.gridcraft import grid_env, mazes

# 1. Setup environment
env = grid_env.GridEnv(mazes.obstacle_maze())

# 2. Create history-conditioned Q-network
network = HistoryConditionedQNetwork(env, history_encoder='lstm', hidden_dim=64)

# 3. Create FQI algorithm
fqi = HistoryConditionedFQI(
    env=env,
    network=network,
    history_length=10,
    jitter_prob=0.5,
    jitter_type='mixed',
    optimal_q_path='fqi_results/maze_with_obstacles/optimal_q.npy',
    use_optimal_q_targets=True,
    lr=1e-3,
    discount=0.99,
    max_project_steps=1000
)

# 4. Train
for i in range(100):
    fqi.update(step=i)
```

## Trajectory Jitter Examples

### Rewind
```python
# Original: [s1, s2, s3, s4, s5]
# Result:   [s1, s2, s3, s4, s5, s4, s3, s2]
# Masks:    [1,  1,  1,  1,  1,  0,  0,  0]

states, actions, gt_times, mask = rewind_trajectory(
    states, actions, 
    rewind_steps=3
)
```

### Pause
```python
# Original: [s1, s2, s3, s4, s5]
# Result:   [s1, s2, s3, s3, s3, s4, s5]
# Masks:    [1,  1,  1,  0,  0,  1,  1]

states, actions, gt_times, mask = pause_trajectory(
    states, actions,
    pause_duration=2
)
```

## Custom Update Rule - Three Ways

### 1. Custom Weights
```python
class MyFQI(HistoryConditionedFQI):
    def compute_weights(self, history_states, history_actions, itr=0):
        # Return weights array [batch_size]
        return custom_weights
```

### 2. Custom Targets
```python
class MyFQI(HistoryConditionedFQI):
    def evaluate_target(self, sample_histories, sample_actions,
                       sample_next_states, sample_rewards,
                       masks=None, ground_truth_timesteps=None, mode=None):
        # Use masks to treat jittered samples differently
        # Use self.optimal_q for Q* targets
        return custom_targets
```

### 3. Custom Loss
```python
class MyFQI(HistoryConditionedFQI):
    def project(self, network=None, optimizer=None, sampler=None):
        # Implement your own training loop
        # Full control over loss computation
        return stopped_mode, loss, num_steps
```

## Key Data Structures

**Masks:** `[batch, history_length]` - 1.0 = real, 0.0 = jittered  
**Ground Truth Timesteps:** `[batch, history_length]` - Original episode timestep  
**History States:** `[batch, history_length]` - State indices for Q-function input

## Loading Optimal Q*

```python
# Option 1: Pass path to HistoryConditionedFQI
fqi = HistoryConditionedFQI(
    optimal_q_path='fqi_results/maze_with_lava/optimal_q.npy',
    use_optimal_q_targets=True
)

# Option 2: Load manually
from debugq.algos.utils import load_optimal_q
optimal_q = load_optimal_q('fqi_results/maze_with_lava/optimal_q.npy')
```

## Available Mazes with Optimal Q*

```
fqi_results/maze_with_lava/optimal_q.npy
fqi_results/maze_with_obstacles/optimal_q.npy
fqi_results/larger_maze/optimal_q.npy
```

## Run Example

```bash
python example_history_fqi.py
```

## Run Tests

```bash
pytest debugq/algos/test_history_fqi.py -v
```

## Parameters to Tune

**History Length:** How many timesteps to condition on (default: 10)
```python
history_length=10
```

**Jitter Probability:** Fraction of samples to augment (default: 0.5)
```python
jitter_prob=0.5
```

**Jitter Type:** 'rewind', 'pause', 'mixed', or 'none'
```python
jitter_type='mixed'
```

**Rewind Steps:** How far to rewind (default: 3)
```python
jitter_kwargs={'rewind_steps': 3}
```

**Pause Duration:** How long to pause (default: 2)
```python
jitter_kwargs={'pause_duration': 2, 'num_pauses': 1}
```

**Network Architecture:** LSTM vs GRU, hidden size
```python
network = HistoryConditionedQNetwork(
    env, 
    history_encoder='lstm',  # or 'gru'
    hidden_dim=64,
    num_layers=1
)
```

## What You Need to Implement

The framework handles:
- ✅ Trajectory collection
- ✅ Replay buffer management
- ✅ History segment sampling
- ✅ Jitter augmentation
- ✅ Mask and timestep tracking
- ✅ Q* loading
- ✅ Basic training loop

You implement:
- ⚠️ **Custom weighting** of jittered vs clean samples
- ⚠️ **Custom target computation** using Q* and masks
- ⚠️ **Custom loss function** (optional)

## Full Documentation

- `HISTORY_FQI_README.md` - Comprehensive guide with examples
- `IMPLEMENTATION_SUMMARY.md` - Detailed changes and customization points
- `example_history_fqi.py` - Complete working example with notes

## Common Patterns

### Store masks for later use
```python
class MyFQI(HistoryConditionedFQI):
    def get_sample_states(self, itr=0):
        result = super().get_sample_states(itr=itr)
        self.current_masks = result[5]  # masks
        self.current_gt_timesteps = result[6]  # timesteps
        return result
    
    def compute_weights(self, history_states, history_actions, itr=0):
        # Now you can access self.current_masks
        is_clean = (self.current_masks.sum(axis=1) == self.history_length)
        weights = np.where(is_clean, 1.0, 0.5)
        return weights / weights.sum()
```

### Different targets for jittered samples
```python
class MyFQI(HistoryConditionedFQI):
    def evaluate_target(self, sample_histories, sample_actions,
                       sample_next_states, sample_rewards,
                       masks=None, ground_truth_timesteps=None, mode=None):
        targets = []
        for i in range(len(sample_next_states)):
            is_clean = (masks[i].sum() == self.history_length)
            
            if is_clean and self.optimal_q is not None:
                # Use Q* for clean samples
                v = q_iteration.logsumexp(self.optimal_q[sample_next_states[i]], 
                                         alpha=self.ent_wt)
            else:
                # Use current Q for jittered samples
                v = q_iteration.logsumexp(self.current_q[sample_next_states[i]], 
                                         alpha=self.ent_wt)
            
            targets.append(sample_rewards[i] + self.discount * v)
        
        return ptu.tensor(targets)
```

## That's It!

You now have a fully functional history-conditioned FQI implementation with trajectory jitter support. Start with `example_history_fqi.py` and customize from there.


