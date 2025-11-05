"""
Unit tests for history-conditioned FQI components.
"""

import numpy as np
import pytest
from debugq.models.q_networks import HistoryConditionedQNetwork
from debugq.algos.history_fqi import HistoryConditionedFQI
from debugq.algos.replay_buffer_fqi import TrajectoryReplayBuffer
from debugq.algos import trajectory_augmentation
from rlutil.envs.gridcraft import grid_env, mazes


def test_trajectory_replay_buffer():
    """Test trajectory replay buffer basic functionality."""
    buffer = TrajectoryReplayBuffer(capacity=1000)
    
    # Add a trajectory
    states = np.array([0, 1, 2, 3, 4])
    actions = np.array([0, 1, 0, 1, 0])
    next_states = np.array([1, 2, 3, 4, 5])
    rewards = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    dones = np.array([False, False, False, False, True])
    
    buffer.add_trajectory(states, actions, next_states, rewards, dones)
    
    assert len(buffer) == 5
    assert len(buffer.trajectories) == 1
    
    # Sample history segments
    history_states, history_actions, ns, r, masks, timesteps = buffer.sample_history_segments(
        batch_size=2, history_length=3
    )
    
    assert history_states.shape == (2, 3)
    assert history_actions.shape == (2, 3)
    assert masks.shape == (2, 3)
    assert timesteps.shape == (2, 3)
    assert len(ns) == 2
    assert len(r) == 2


def test_trajectory_augmentation_rewind():
    """Test rewind augmentation."""
    states = np.array([0, 1, 2, 3, 4])
    actions = np.array([0, 1, 0, 1, 0])
    
    aug_states, aug_actions, gt_times, mask = trajectory_augmentation.rewind_trajectory(
        states, actions, rewind_steps=2, rewind_prob=1.0
    )
    
    # Should have original 5 + 2 rewound = 7
    assert len(aug_states) == 7
    assert len(mask) == 7
    
    # First 5 should be original
    assert np.all(mask[:5] == 1.0)
    # Last 2 should be rewound
    assert np.all(mask[5:] == 0.0)
    
    # Check rewind correctness
    assert aug_states[5] == states[3]
    assert aug_states[6] == states[2]


def test_trajectory_augmentation_pause():
    """Test pause augmentation."""
    states = np.array([0, 1, 2, 3, 4])
    actions = np.array([0, 1, 0, 1, 0])
    
    aug_states, aug_actions, gt_times, mask = trajectory_augmentation.pause_trajectory(
        states, actions, pause_duration=2, num_pauses=1, pause_prob=1.0
    )
    
    # Should have original 5 + 2 paused = 7
    assert len(aug_states) == 7
    assert len(mask) == 7
    
    # Should have 5 original timesteps and 2 paused
    assert mask.sum() == 5
    assert (mask == 0).sum() == 2


def test_history_network_forward():
    """Test history-conditioned network forward pass."""
    env = grid_env.GridEnv(mazes.simple_maze())
    
    network = HistoryConditionedQNetwork(
        env,
        history_encoder='lstm',
        hidden_dim=32,
        num_layers=1
    )
    
    # Create batch of history sequences
    batch_size = 4
    history_length = 5
    history_states = np.random.randint(0, env.num_states, size=(batch_size, history_length))
    masks = np.ones((batch_size, history_length), dtype=np.float32)
    
    import torch
    history_states = torch.tensor(history_states, dtype=torch.int64)
    masks = torch.tensor(masks, dtype=torch.float32)
    
    # Forward pass
    q_values = network(history_states, mask=masks)
    
    assert q_values.shape == (batch_size, env.num_actions)


def test_history_network_with_mask():
    """Test that network properly uses mask to select last valid timestep."""
    env = grid_env.GridEnv(mazes.simple_maze())
    
    network = HistoryConditionedQNetwork(
        env,
        history_encoder='lstm',
        hidden_dim=32,
        num_layers=1
    )
    
    import torch
    
    # Create a sequence with padding at the start
    history_length = 5
    # First 2 are padding, last 3 are valid
    history_states = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.int64)
    masks = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.float32)
    
    q_values = network(history_states, mask=masks)
    
    assert q_values.shape == (1, env.num_actions)


def test_history_fqi_initialization():
    """Test that HistoryConditionedFQI initializes correctly."""
    env = grid_env.GridEnv(mazes.simple_maze())
    
    network = HistoryConditionedQNetwork(
        env,
        history_encoder='lstm',
        hidden_dim=32,
        num_layers=1
    )
    
    fqi = HistoryConditionedFQI(
        env=env,
        network=network,
        history_length=5,
        batch_size=8,
        jitter_prob=0.5,
        jitter_type='mixed',
        lr=1e-3,
        discount=0.99,
        max_project_steps=10,
        min_project_steps=1,
        backup_mode='exact'
    )
    
    assert fqi.history_length == 5
    assert fqi.batch_size == 8
    assert fqi.jitter_prob == 0.5
    assert isinstance(fqi.replay_buffer, TrajectoryReplayBuffer)


def test_history_fqi_sample_generation():
    """Test that get_sample_states works correctly."""
    env = grid_env.GridEnv(mazes.simple_maze())
    
    network = HistoryConditionedQNetwork(
        env,
        history_encoder='lstm',
        hidden_dim=32,
        num_layers=1
    )
    
    fqi = HistoryConditionedFQI(
        env=env,
        network=network,
        history_length=5,
        batch_size=4,
        jitter_prob=0.0,  # No jitter for this test
        lr=1e-3,
        discount=0.99,
        max_project_steps=10,
        min_project_steps=1,
        backup_mode='exact'
    )
    
    # Add some trajectories to the buffer
    for _ in range(3):
        states = np.random.randint(0, env.num_states, size=10)
        actions = np.random.randint(0, env.num_actions, size=10)
        next_states = np.random.randint(0, env.num_states, size=10)
        rewards = np.random.rand(10)
        dones = np.zeros(10, dtype=bool)
        dones[-1] = True
        
        fqi.replay_buffer.add_trajectory(states, actions, next_states, rewards, dones)
    
    # Sample
    result = fqi.get_sample_states(itr=0)
    
    assert len(result) == 7  # history_states, actions, next_states, rewards, weights, masks, timesteps
    
    history_states, history_actions, next_states, rewards, weights, masks, timesteps = result
    
    assert history_states.shape == (4, 5)
    assert history_actions.shape == (4, 5)
    assert len(next_states) == 4
    assert len(rewards) == 4
    assert len(weights) == 4
    assert masks.shape == (4, 5)
    assert timesteps.shape == (4, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


