"""
Trajectory augmentation for history-conditioned Q-learning.

This module provides functions to create "rewound" and "paused" trajectories
that simulate temporal jitter for training history-conditioned Q-functions.
"""

import numpy as np


def rewind_trajectory(states, actions, rewind_steps=3, rewind_prob=1.0):
    """
    Create a rewound trajectory by appending reversed history.
    
    Example: [s1, s2, s3, s4, s5] with rewind_steps=3 
             -> [s1, s2, s3, s4, s5, s4, s3, s2]
    
    Args:
        states: Array of state indices [T]
        actions: Array of actions [T]
        rewind_steps: Number of steps to rewind
        rewind_prob: Probability of applying rewind
    
    Returns:
        augmented_states: Extended state sequence
        augmented_actions: Extended action sequence
        ground_truth_timesteps: Original timestep for each position [T_aug]
        mask: Binary mask indicating original (1) vs rewound (0) timesteps [T_aug]
    """
    if np.random.rand() > rewind_prob or len(states) <= rewind_steps:
        # No rewind, return original
        T = len(states)
        return (states, actions, 
                np.arange(T), 
                np.ones(T, dtype=np.float32))
    
    T = len(states)
    rewind_steps = min(rewind_steps, T - 1)
    
    # Take the last rewind_steps states and reverse them (excluding the last state itself)
    rewind_states = states[-rewind_steps-1:-1][::-1]
    rewind_actions = actions[-rewind_steps-1:-1][::-1]
    
    # Concatenate original + rewound
    augmented_states = np.concatenate([states, rewind_states])
    augmented_actions = np.concatenate([actions, rewind_actions])
    
    # Ground truth timesteps
    original_timesteps = np.arange(T)
    rewind_timesteps = np.arange(T - 2, T - 2 - rewind_steps, -1)
    ground_truth_timesteps = np.concatenate([original_timesteps, rewind_timesteps])
    
    # Mask: 1 for original, 0 for rewound
    mask = np.concatenate([
        np.ones(T, dtype=np.float32),
        np.zeros(rewind_steps, dtype=np.float32)
    ])
    
    return augmented_states, augmented_actions, ground_truth_timesteps, mask


def pause_trajectory(states, actions, pause_duration=3, num_pauses=1, pause_prob=1.0):
    """
    Create a paused trajectory by repeating states at certain positions.
    
    Example: [s1, s2, s3, s4, s5] with pause at position 2, duration 2
             -> [s1, s2, s3, s3, s3, s4, s5]
    
    Args:
        states: Array of state indices [T]
        actions: Array of actions [T]
        pause_duration: Number of times to repeat the paused state
        num_pauses: Number of pause points to insert
        pause_prob: Probability of applying pause
    
    Returns:
        augmented_states: Extended state sequence
        augmented_actions: Extended action sequence
        ground_truth_timesteps: Original timestep for each position [T_aug]
        mask: Binary mask indicating original (1) vs paused (0) timesteps [T_aug]
    """
    if np.random.rand() > pause_prob or len(states) <= num_pauses:
        # No pause, return original
        T = len(states)
        return (states, actions,
                np.arange(T),
                np.ones(T, dtype=np.float32))
    
    T = len(states)
    
    # Select random positions to pause (excluding first and last)
    if T <= 2:
        pause_positions = []
    else:
        pause_positions = np.random.choice(
            np.arange(1, T - 1), 
            size=min(num_pauses, T - 2), 
            replace=False
        )
        pause_positions = np.sort(pause_positions)
    
    if len(pause_positions) == 0:
        # No valid pause positions
        return (states, actions,
                np.arange(T),
                np.ones(T, dtype=np.float32))
    
    # Build augmented trajectory
    augmented_states = []
    augmented_actions = []
    ground_truth_timesteps = []
    mask = []
    
    current_idx = 0
    for pause_pos in pause_positions:
        # Add states up to pause position
        augmented_states.extend(states[current_idx:pause_pos+1])
        augmented_actions.extend(actions[current_idx:pause_pos+1])
        ground_truth_timesteps.extend(range(current_idx, pause_pos+1))
        mask.extend([1.0] * (pause_pos + 1 - current_idx))
        
        # Add paused repetitions
        for _ in range(pause_duration):
            augmented_states.append(states[pause_pos])
            augmented_actions.append(actions[pause_pos])
            ground_truth_timesteps.append(pause_pos)
            mask.append(0.0)
        
        current_idx = pause_pos + 1
    
    # Add remaining states
    if current_idx < T:
        augmented_states.extend(states[current_idx:])
        augmented_actions.extend(actions[current_idx:])
        ground_truth_timesteps.extend(range(current_idx, T))
        mask.extend([1.0] * (T - current_idx))
    
    return (np.array(augmented_states), 
            np.array(augmented_actions),
            np.array(ground_truth_timesteps),
            np.array(mask, dtype=np.float32))


def apply_jitter(states, actions, jitter_type='mixed', **kwargs):
    """
    Apply random jitter to trajectory (rewind, pause, or mixed).
    
    Args:
        states: Array of state indices [T]
        actions: Array of actions [T]
        jitter_type: Type of jitter - 'rewind', 'pause', 'mixed', or 'none'
        **kwargs: Additional arguments for specific jitter functions
    
    Returns:
        augmented_states: Extended state sequence
        augmented_actions: Extended action sequence
        ground_truth_timesteps: Original timestep for each position [T_aug]
        mask: Binary mask indicating original (1) vs jittered (0) timesteps [T_aug]
    """
    if jitter_type == 'none':
        T = len(states)
        return (states, actions,
                np.arange(T),
                np.ones(T, dtype=np.float32))
    
    elif jitter_type == 'rewind':
        return rewind_trajectory(states, actions, **kwargs)
    
    elif jitter_type == 'pause':
        return pause_trajectory(states, actions, **kwargs)
    
    elif jitter_type == 'mixed':
        # Randomly choose between rewind and pause
        if np.random.rand() < 0.5:
            return rewind_trajectory(states, actions, **kwargs)
        else:
            return pause_trajectory(states, actions, **kwargs)
    
    else:
        raise ValueError(f"Unknown jitter type: {jitter_type}")


def extract_history_segments(states, actions, segment_length, stride=1):
    """
    Extract overlapping history segments from a trajectory.
    
    Args:
        states: Array of state indices [T]
        actions: Array of actions [T]
        segment_length: Length of each history segment
        stride: Stride between segments
    
    Returns:
        history_states: Array of shape [N, segment_length]
        history_actions: Array of shape [N, segment_length]
        final_states: The last state in each segment [N]
        final_actions: The action taken at the last state [N]
    """
    T = len(states)
    if T < segment_length:
        # Pad with the first state if trajectory is too short
        pad_length = segment_length - T
        padded_states = np.concatenate([
            np.repeat(states[0], pad_length),
            states
        ])
        padded_actions = np.concatenate([
            np.repeat(actions[0], pad_length),
            actions
        ])
        return (padded_states.reshape(1, -1),
                padded_actions.reshape(1, -1),
                np.array([states[-1]]),
                np.array([actions[-1]]))
    
    # Extract segments
    num_segments = (T - segment_length) // stride + 1
    history_states = []
    history_actions = []
    final_states = []
    final_actions = []
    
    for i in range(num_segments):
        start_idx = i * stride
        end_idx = start_idx + segment_length
        
        if end_idx <= T:
            history_states.append(states[start_idx:end_idx])
            history_actions.append(actions[start_idx:end_idx])
            final_states.append(states[end_idx - 1])
            final_actions.append(actions[end_idx - 1])
    
    return (np.array(history_states),
            np.array(history_actions),
            np.array(final_states),
            np.array(final_actions))


def create_jittered_dataset(trajectories, history_length, jitter_prob=0.5, 
                           jitter_type='mixed', **jitter_kwargs):
    """
    Create a dataset of jittered history segments from trajectories.
    
    Args:
        trajectories: List of (states, actions, rewards, dones) tuples
        history_length: Length of history to extract
        jitter_prob: Probability of applying jitter to each trajectory
        jitter_type: Type of jitter to apply
        **jitter_kwargs: Additional arguments for jitter functions
    
    Returns:
        Dictionary containing:
            - history_states: [N, history_length]
            - history_actions: [N, history_length]
            - ground_truth_timesteps: [N, history_length]
            - masks: [N, history_length]
            - final_actions: [N]
    """
    all_history_states = []
    all_history_actions = []
    all_gt_timesteps = []
    all_masks = []
    all_final_actions = []
    
    for states, actions, rewards, dones in trajectories:
        # Apply jitter with probability jitter_prob
        if np.random.rand() < jitter_prob:
            aug_states, aug_actions, gt_times, mask = apply_jitter(
                states, actions, jitter_type=jitter_type, **jitter_kwargs
            )
        else:
            T = len(states)
            aug_states, aug_actions = states, actions
            gt_times = np.arange(T)
            mask = np.ones(T, dtype=np.float32)
        
        # Extract history segments
        T_aug = len(aug_states)
        for i in range(history_length - 1, T_aug):
            # History is from [i - history_length + 1 : i + 1]
            start_idx = i - history_length + 1
            
            history_seg_states = aug_states[start_idx:i+1]
            history_seg_actions = aug_actions[start_idx:i+1]
            history_gt_times = gt_times[start_idx:i+1]
            history_mask = mask[start_idx:i+1]
            
            all_history_states.append(history_seg_states)
            all_history_actions.append(history_seg_actions)
            all_gt_timesteps.append(history_gt_times)
            all_masks.append(history_mask)
            all_final_actions.append(aug_actions[i])
    
    return {
        'history_states': np.array(all_history_states),
        'history_actions': np.array(all_history_actions),
        'ground_truth_timesteps': np.array(all_gt_timesteps),
        'masks': np.array(all_masks),
        'final_actions': np.array(all_final_actions)
    }


