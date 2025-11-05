import numpy as np
import os

from rlutil.envs.tabular import q_iteration

def run_rollout(env, q_fn, ent_wt=1.0, render=False):
    s0 = env.reset_state()
    states = []
    actions = []
    next_states = []
    pol = q_iteration.get_policy(q_fn, ent_wt=ent_wt)

    rewards = []
    while True:
        states.append(env.get_state())
        probs = pol[env.get_state()]
        action = np.random.choice(np.arange(0, env.num_actions), p=probs)
        ts = env.step_state(action)
        rewards.append(ts['reward'])
        actions.append(action)
        next_states.append(env.get_state())
        if ts['done']:
            break
        if render:
            env.render()
    return states, actions, next_states, rewards


def eval_policy_qfn(env, q_fn, n_rollout=10, ent_wt=1.0, render=False):
    returns = []
    for i in range(n_rollout):
        _, _, _, rews = run_rollout(env, q_fn, ent_wt=ent_wt,
            render=render and (i==n_rollout-1))
        returns.append(np.sum(rews))
    return np.mean(returns)


def load_optimal_q(path):
    """
    Load optimal Q-function from saved results.
    
    Args:
        path: Path to .npy or .npz file containing Q-values
    
    Returns:
        Q-values as numpy array of shape [num_states, num_actions]
    
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Q-function file not found: {path}")
    
    if path.endswith('.npy'):
        q_values = np.load(path)
    elif path.endswith('.npz'):
        data = np.load(path)
        # Try common keys
        if 'optimal_q' in data:
            q_values = data['optimal_q']
        elif 'q_values' in data:
            q_values = data['q_values']
        elif 'arr_0' in data:
            q_values = data['arr_0']
        else:
            raise ValueError(f"Could not find Q-values in .npz file. Available keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unknown file format: {path}. Supported formats: .npy, .npz")
    
    return q_values


def collect_trajectories(env, q_fn, num_trajectories, ent_wt=1.0, max_length=None):
    """
    Collect trajectories by rolling out a policy.
    
    Args:
        env: Environment
        q_fn: Q-function to derive policy from
        num_trajectories: Number of trajectories to collect
        ent_wt: Entropy weight for policy
        max_length: Maximum trajectory length (None for no limit)
    
    Returns:
        List of (states, actions, next_states, rewards, dones) tuples
    """
    trajectories = []
    
    for _ in range(num_trajectories):
        states, actions, next_states, rewards = run_rollout(env, q_fn, ent_wt=ent_wt)
        
        # Create done flags (all False except last)
        dones = np.zeros(len(states), dtype=bool)
        dones[-1] = True
        
        # Truncate if max_length specified
        if max_length is not None and len(states) > max_length:
            states = np.array(states[:max_length])
            actions = np.array(actions[:max_length])
            next_states = np.array(next_states[:max_length])
            rewards = np.array(rewards[:max_length])
            dones = np.zeros(max_length, dtype=bool)
            dones[-1] = True
        else:
            states = np.array(states)
            actions = np.array(actions)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
        
        trajectories.append((states, actions, next_states, rewards, dones))
    
    return trajectories
