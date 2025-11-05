# Architecture Diagram: History-Conditioned FQI

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   History-Conditioned FQI System                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │   HistoryConditionedFQI Algorithm    │
        │  (debugq/algos/history_fqi.py)       │
        └──────────────────────────────────────┘
                     │         │         │
        ┌────────────┘         │         └────────────┐
        ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Q-Network    │    │ Replay Buffer    │    │ Jitter           │
│ (LSTM/GRU)   │    │ (Trajectories)   │    │ Augmentation     │
└──────────────┘    └──────────────────┘    └──────────────────┘
```

## Detailed Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Collect Trajectories                                        │
└─────────────────────────────────────────────────────────────────────┘
    Environment + Current Policy
              │
              ▼
    [s₀, s₁, s₂, ..., sₜ]  ← Full trajectory
    [a₀, a₁, a₂, ..., aₜ]
    [r₀, r₁, r₂, ..., rₜ]
              │
              ▼
    TrajectoryReplayBuffer.add_trajectory()

┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Sample History Segments                                     │
└─────────────────────────────────────────────────────────────────────┘
    TrajectoryReplayBuffer.sample_history_segments(
        batch_size=32, 
        history_length=10
    )
              │
              ▼
    history_states:  [batch=32, history_length=10]
    next_states:     [batch=32]
    rewards:         [batch=32]
    masks:           [batch=32, history_length=10]
    timesteps:       [batch=32, history_length=10]

┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Apply Jitter Augmentation (Optional, prob=0.5)             │
└─────────────────────────────────────────────────────────────────────┘
    Original: [s₁, s₂, s₃, s₄, s₅]
              │
              ├─→ Rewind: [s₁, s₂, s₃, s₄, s₅, s₄, s₃, s₂]
              │           mask: [1,  1,  1,  1,  1,  0,  0,  0]
              │
              └─→ Pause:  [s₁, s₂, s₃, s₃, s₃, s₄, s₅]
                          mask: [1,  1,  1,  0,  0,  1,  1]

┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Compute Q-values (Forward Pass)                             │
└─────────────────────────────────────────────────────────────────────┘
    history_states [batch, history_length]
              │
              ▼
    Lookup observations [batch, history_length, obs_dim]
              │
              ▼
    LSTM/GRU Encoder [batch, history_length, hidden_dim]
              │
              ▼
    Select last valid timestep (using mask)
              │
              ▼
    MLP Head [batch, hidden_dim] → [batch, num_actions]
              │
              ▼
    Q-values [batch, num_actions]

┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Compute Target Q-values                                     │
└─────────────────────────────────────────────────────────────────────┘
    Option A: Use Optimal Q*
    ────────────────────────
    Load Q* from file
              │
              ▼
    V*(s') = logsumexp(Q*[s'], α)
              │
              ▼
    target = r + γ · V*(s')

    Option B: Use Current Q
    ───────────────────────
    V(s') = logsumexp(Q[s'], α)
              │
              ▼
    target = r + γ · V(s')

    Option C: Custom (Your Implementation)
    ───────────────────────────────────────
    Use masks to identify jittered samples
              │
              ▼
    if jittered:
        target = custom_logic_1(...)
    else:
        target = custom_logic_2(...)

┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: Compute Loss with Custom Weights                            │
└─────────────────────────────────────────────────────────────────────┘
    Q-values [batch, num_actions]
    Targets  [batch, num_actions]
    Weights  [batch]  ← YOUR CUSTOM WEIGHTING
              │
              ▼
    Loss = Σᵢ weights[i] · mean((Q[i] - target[i])²)
              │
              ▼
    Backprop & Update

┌─────────────────────────────────────────────────────────────────────┐
│ Step 7: Update Current Q-function                                   │
└─────────────────────────────────────────────────────────────────────┘
    Extract Q-values for all states
              │
              ▼
    current_Q [num_states, num_actions]
              │
              ▼
    Use for next iteration's policy
```

## Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    User's Custom Class                           │
│    class MyHistoryFQI(HistoryConditionedFQI):                    │
│        def compute_weights(...)        ← CUSTOMIZE               │
│        def evaluate_target(...)        ← CUSTOMIZE               │
│        def project(...)                ← CUSTOMIZE (optional)    │
└──────────────────────────────────────────────────────────────────┘
                              │ inherits from
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              HistoryConditionedFQI (Base Class)                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ update():                                              │     │
│  │   1. pre_project()      ← Collect trajectories        │     │
│  │   2. project()          ← Train network               │     │
│  │   3. post_project()     ← Logging/validation          │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Uses:                                                           │
│  ├─→ TrajectoryReplayBuffer (store & sample)                    │
│  ├─→ trajectory_augmentation (rewind/pause)                     │
│  ├─→ HistoryConditionedQNetwork (Q-function)                    │
│  └─→ load_optimal_q() (optional Q* loading)                     │
└──────────────────────────────────────────────────────────────────┘
                              │ inherits from
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FQI (Base Class)                             │
│  Standard FQI components:                                        │
│  - Backup computation                                            │
│  - Policy evaluation                                             │
│  - Logging utilities                                             │
└──────────────────────────────────────────────────────────────────┘
```

## Customization Extension Points

```
┌────────────────────────────────────────────────────────────────────┐
│ Method Override                 │ Purpose                          │
├─────────────────────────────────┼──────────────────────────────────┤
│ compute_weights()               │ Weight samples (clean vs jitter) │
│   Input: history_states, masks  │   → weights [batch]              │
├─────────────────────────────────┼──────────────────────────────────┤
│ evaluate_target()               │ Compute target Q-values          │
│   Input: histories, masks,      │   → targets [batch, actions]     │
│          gt_timesteps, Q*       │                                  │
├─────────────────────────────────┼──────────────────────────────────┤
│ get_sample_states()             │ Modify sampling behavior         │
│   Can: filter, reweight,        │   → (histories, masks, ...)      │
│        store masks              │                                  │
├─────────────────────────────────┼──────────────────────────────────┤
│ project()                       │ Full training loop control       │
│   Can: custom loss, different   │   → (stopped_mode, loss, steps)  │
│        optimization             │                                  │
└─────────────────────────────────┴──────────────────────────────────┘
```

## Mask and Ground Truth Timestep Example

```
Original Trajectory:
┌────┬────┬────┬────┬────┐
│ s₁ │ s₂ │ s₃ │ s₄ │ s₅ │
└────┴────┴────┴────┴────┘
  t=0  t=1  t=2  t=3  t=4

After Rewind (steps=2):
┌────┬────┬────┬────┬────┬────┬────┐
│ s₁ │ s₂ │ s₃ │ s₄ │ s₅ │ s₄ │ s₃ │  ← states
└────┴────┴────┴────┴────┴────┴────┘
  1    1    1    1    1    0    0     ← masks (1=real, 0=jitter)
  0    1    2    3    4    3    2     ← ground truth timesteps

After Pause (at position 2, duration=2):
┌────┬────┬────┬────┬────┬────┬────┐
│ s₁ │ s₂ │ s₃ │ s₃ │ s₃ │ s₄ │ s₅ │  ← states
└────┴────┴────┴────┴────┴────┴────┘
  1    1    1    0    0    1    1     ← masks
  0    1    2    2    2    3    4     ← ground truth timesteps
                ↑────↑
                paused repetitions
```

## Network Architecture Detail

```
HistoryConditionedQNetwork
─────────────────────────────

Input Layer:
  history_states: [batch, history_length]
        ↓
  Index lookup into all_observations
        ↓
  observations: [batch, history_length, obs_dim]

Encoder:
  LSTM (batch_first=True):
    input_size = obs_dim
    hidden_size = 64
    num_layers = 1
        ↓
  output: [batch, history_length, 64]
  hidden: [1, batch, 64]

Mask Handling:
  If mask provided:
    lengths = mask.sum(dim=1) - 1
    final_hidden = output[batch_idx, lengths]
  Else:
    final_hidden = output[:, -1, :]
        ↓
  final_hidden: [batch, 64]

Output Head:
  Linear(64, 64) → ReLU → Linear(64, num_actions)
        ↓
  q_values: [batch, num_actions]
```

## File Structure Summary

```
debugq/
│
├── models/
│   └── q_networks.py
│       ├── TabularNetwork         (existing)
│       ├── LinearNetwork          (existing)
│       ├── FCNetwork              (existing)
│       └── HistoryConditionedQNetwork  ← NEW
│
├── algos/
│   ├── fqi.py                     (existing base class)
│   ├── exact_fqi.py               (existing)
│   ├── sampling_fqi.py            (existing)
│   │
│   ├── replay_buffer_fqi.py
│   │   ├── ReplayBuffer           (existing)
│   │   ├── SimpleReplayBuffer     (existing)
│   │   ├── TabularReplayBuffer    (existing)
│   │   └── TrajectoryReplayBuffer ← NEW
│   │
│   ├── utils.py
│   │   ├── run_rollout()          (existing)
│   │   ├── eval_policy_qfn()      (existing)
│   │   ├── load_optimal_q()       ← NEW
│   │   └── collect_trajectories() ← NEW
│   │
│   ├── trajectory_augmentation.py ← NEW
│   │   ├── rewind_trajectory()
│   │   ├── pause_trajectory()
│   │   ├── apply_jitter()
│   │   └── create_jittered_dataset()
│   │
│   ├── history_fqi.py             ← NEW
│   │   └── HistoryConditionedFQI
│   │       ├── pre_project()
│   │       ├── get_sample_states()
│   │       ├── compute_weights()        ← CUSTOMIZE
│   │       ├── evaluate_qvalues()
│   │       ├── evaluate_target()        ← CUSTOMIZE
│   │       ├── project()                ← CUSTOMIZE
│   │       └── eval_policy()
│   │
│   └── test_history_fqi.py        ← NEW (unit tests)
│
example_history_fqi.py             ← NEW (working example)
HISTORY_FQI_README.md              ← NEW (full docs)
IMPLEMENTATION_SUMMARY.md          ← NEW (what changed)
QUICKSTART.md                      ← NEW (quick reference)
ARCHITECTURE_DIAGRAM.md            ← NEW (this file)
```

## Data Dimensions Reference

```
Environment:
  num_states:   S
  num_actions:  A
  obs_dim:      D

Batch Parameters:
  batch_size:       B
  history_length:   H

Arrays:
  history_states:         [B, H]     (state indices)
  history_actions:        [B, H]     (action indices)
  observations:           [B, H, D]  (actual observations)
  masks:                  [B, H]     (validity: 0 or 1)
  ground_truth_timesteps: [B, H]     (original timestep indices)
  next_states:            [B]        (next state after history)
  rewards:                [B]        (reward for last action)
  weights:                [B]        (sample weights)
  
  Q-values:               [B, A]     (Q(history, a) for all actions)
  targets:                [B, A]     (target Q-values)
  
  optimal_q:              [S, A]     (loaded from file)
  current_q:              [S, A]     (learned Q-function)
```


