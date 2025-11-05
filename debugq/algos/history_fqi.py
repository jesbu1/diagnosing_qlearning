"""
History-Conditioned Fitted Q-Iteration.

This module implements FQI algorithms that train Q-functions conditioned on
arbitrary-length observation histories, with support for trajectory jitter
(rewinding and pausing) to simulate temporal perturbations.
"""

import numpy as np
import torch
import six

from rlutil.logging import logger, hyperparameterized
from rlutil.envs.tabular import q_iteration

from debugq.algos import fqi, replay_buffer_fqi, utils, trajectory_augmentation
import debugq.pytorch_util as ptu


@six.add_metaclass(hyperparameterized.Hyperparameterized)
class HistoryConditionedFQI(fqi.FQI):
    """
    FQI with a history-conditioned Q-function.
    
    The Q-function takes as input a sequence of observations (history) and outputs
    Q-values for the current state. Supports trajectory jitter augmentation.
    """
    
    def __init__(self, env, network, 
                 history_length=10,
                 replay_buffer=None,
                 batch_size=32,
                 jitter_prob=0.5,
                 jitter_type='mixed',
                 jitter_kwargs=None,
                 optimal_q_path=None,
                 use_optimal_q_targets=False,
                 collect_trajectories_per_iter=10,
                 **kwargs):
        """
        Args:
            env: Environment
            network: HistoryConditionedQNetwork instance
            history_length: Length of observation history to condition on
            replay_buffer: TrajectoryReplayBuffer instance (created if None)
            batch_size: Batch size for training
            jitter_prob: Probability of applying jitter augmentation
            jitter_type: Type of jitter ('rewind', 'pause', 'mixed', 'none')
            jitter_kwargs: Additional kwargs for jitter functions
            optimal_q_path: Path to load optimal Q* for target computation
            use_optimal_q_targets: If True, use loaded Q* for all targets
            collect_trajectories_per_iter: Number of trajectories to collect per iteration
            **kwargs: Additional arguments for base FQI
        """
        super(HistoryConditionedFQI, self).__init__(env, network, **kwargs)
        
        self.history_length = history_length
        self.batch_size = batch_size
        self.jitter_prob = jitter_prob
        self.jitter_type = jitter_type
        self.jitter_kwargs = jitter_kwargs or {}
        self.collect_trajectories_per_iter = collect_trajectories_per_iter
        
        # Set up replay buffer
        if replay_buffer is None:
            self.replay_buffer = replay_buffer_fqi.TrajectoryReplayBuffer(capacity=100000)
        else:
            self.replay_buffer = replay_buffer
        
        # Load optimal Q* if provided
        self.use_optimal_q_targets = use_optimal_q_targets
        if optimal_q_path is not None:
            logger.log(f"Loading optimal Q* from {optimal_q_path}")
            self.optimal_q = utils.load_optimal_q(optimal_q_path)
            logger.log(f"Loaded Q* with shape {self.optimal_q.shape}")
        else:
            self.optimal_q = None
        
        # Storage for current trajectories
        self.current_trajectories = []
    
    def pre_project(self):
        """Collect new trajectories and add to replay buffer."""
        # Collect trajectories using current policy
        trajectories = utils.collect_trajectories(
            self.env, 
            self.current_q,  # Use current Q for policy
            num_trajectories=self.collect_trajectories_per_iter,
            ent_wt=self.ent_wt
        )
        
        # Add trajectories to replay buffer
        for states, actions, next_states, rewards, dones in trajectories:
            self.replay_buffer.add_trajectory(states, actions, next_states, rewards, dones)
        
        logger.record_tabular('replay_buffer_size', len(self.replay_buffer))
        logger.record_tabular('num_trajectories', len(self.replay_buffer.trajectories))
    
    def get_sample_states(self, itr=0):
        """
        Sample history segments from replay buffer with optional jitter augmentation.
        
        Returns:
            Tuple of:
                - history_states: [batch_size, history_length]
                - history_actions: [batch_size, history_length] (not used for Q-value, but returned for consistency)
                - next_states: [batch_size]
                - rewards: [batch_size]
                - weights: [batch_size]
                - masks: [batch_size, history_length]
                - ground_truth_timesteps: [batch_size, history_length]
        """
        if len(self.replay_buffer) == 0:
            # Buffer is empty, return dummy data
            # This can happen on the first iteration
            logger.log("Warning: Replay buffer is empty, returning dummy samples")
            history_states = np.zeros((self.batch_size, self.history_length), dtype=np.int32)
            history_actions = np.zeros((self.batch_size, self.history_length), dtype=np.int32)
            next_states = np.zeros(self.batch_size, dtype=np.int32)
            rewards = np.zeros(self.batch_size, dtype=np.float32)
            weights = np.ones(self.batch_size, dtype=np.float32)
            masks = np.ones((self.batch_size, self.history_length), dtype=np.float32)
            ground_truth_timesteps = np.zeros((self.batch_size, self.history_length), dtype=np.int32)
            
            return (history_states, history_actions, next_states, rewards, 
                    weights, masks, ground_truth_timesteps)
        
        # Sample trajectories
        (history_states, history_actions, next_states, rewards, 
         masks, timesteps) = self.replay_buffer.sample_history_segments(
            batch_size=self.batch_size,
            history_length=self.history_length
        )
        
        # Apply jitter augmentation
        if self.jitter_prob > 0 and self.jitter_type != 'none':
            history_states, history_actions, masks, timesteps = self._apply_jitter_batch(
                history_states, history_actions, masks, timesteps
            )
        
        # Compute sample weights (uniform for now)
        weights = self.compute_weights(history_states, history_actions, itr=itr)
        
        return (history_states, history_actions, next_states, rewards,
                weights, masks, timesteps)
    
    def _apply_jitter_batch(self, history_states, history_actions, masks, timesteps):
        """
        Apply jitter augmentation to a batch of history segments.
        
        Args:
            history_states: [batch_size, history_length]
            history_actions: [batch_size, history_length]
            masks: [batch_size, history_length]
            timesteps: [batch_size, history_length]
        
        Returns:
            Augmented versions with same shape
        """
        batch_size = history_states.shape[0]
        
        augmented_states = []
        augmented_actions = []
        augmented_masks = []
        augmented_timesteps = []
        
        for i in range(batch_size):
            # Apply jitter with probability
            if np.random.rand() < self.jitter_prob:
                # Get the valid portion of the history (using mask)
                valid_length = int(masks[i].sum())
                if valid_length == 0:
                    valid_length = 1
                
                valid_states = history_states[i, -valid_length:]
                valid_actions = history_actions[i, -valid_length:]
                
                # Apply jitter
                aug_states, aug_actions, aug_timesteps, aug_mask = trajectory_augmentation.apply_jitter(
                    valid_states, valid_actions, 
                    jitter_type=self.jitter_type,
                    **self.jitter_kwargs
                )
                
                # Truncate or pad to history_length
                aug_length = len(aug_states)
                if aug_length >= self.history_length:
                    # Take the last history_length steps
                    aug_states = aug_states[-self.history_length:]
                    aug_actions = aug_actions[-self.history_length:]
                    aug_mask = aug_mask[-self.history_length:]
                    aug_timesteps = aug_timesteps[-self.history_length:]
                else:
                    # Pad at the beginning
                    pad_length = self.history_length - aug_length
                    aug_states = np.concatenate([
                        np.repeat(aug_states[0], pad_length),
                        aug_states
                    ])
                    aug_actions = np.concatenate([
                        np.repeat(aug_actions[0], pad_length),
                        aug_actions
                    ])
                    aug_mask = np.concatenate([
                        np.zeros(pad_length, dtype=np.float32),
                        aug_mask
                    ])
                    aug_timesteps = np.concatenate([
                        np.zeros(pad_length, dtype=np.int32),
                        aug_timesteps
                    ])
                
                augmented_states.append(aug_states)
                augmented_actions.append(aug_actions)
                augmented_masks.append(aug_mask)
                augmented_timesteps.append(aug_timesteps)
            else:
                # Keep original
                augmented_states.append(history_states[i])
                augmented_actions.append(history_actions[i])
                augmented_masks.append(masks[i])
                augmented_timesteps.append(timesteps[i])
        
        return (np.array(augmented_states),
                np.array(augmented_actions),
                np.array(augmented_masks),
                np.array(augmented_timesteps))
    
    def compute_weights(self, history_states, history_actions, itr=0):
        """
        Compute importance weights for samples.
        Override this method to implement custom weighting schemes.
        
        Args:
            history_states: [batch_size, history_length]
            history_actions: [batch_size, history_length]
            itr: Current training iteration
        
        Returns:
            weights: [batch_size]
        """
        return np.ones(history_states.shape[0], dtype=np.float32)
    
    def evaluate_qvalues(self, sample_histories, sample_actions_history, masks=None, 
                        mode=None, network=None):
        """
        Evaluate Q-values for history-conditioned network.
        
        Args:
            sample_histories: [batch_size, history_length] state indices
            sample_actions_history: [batch_size, history_length] action indices (not used by network)
            masks: [batch_size, history_length] validity mask
            mode: Evaluation mode (unused for history networks)
            network: Network to use (defaults to self.network)
        
        Returns:
            Q-values: [batch_size, num_actions] for MULTIPLE_HEADS mode
                     or [batch_size] for FLAT mode
        """
        if network is None:
            network = self.network
        
        # Convert to tensors
        sample_histories = ptu.tensor(sample_histories, dtype=torch.int64)
        if masks is not None:
            masks = ptu.tensor(masks, dtype=torch.float32)
        
        # Forward pass through history-conditioned network
        q_values = network(sample_histories, mask=masks)  # [batch_size, num_actions]
        
        # Return based on mode
        if mode is None:
            mode = self.q_format
        
        if mode == fqi.MULTIPLE_HEADS:
            return q_values
        else:
            # For FLAT mode, need to select specific actions
            # This is handled in the projection loop
            return q_values
    
    def evaluate_target(self, sample_histories, sample_actions, 
                       sample_next_states, sample_rewards,
                       masks=None, ground_truth_timesteps=None, mode=None):
        """
        Evaluate target Q-values.
        
        Args:
            sample_histories: [batch_size, history_length] state indices
            sample_actions: [batch_size, history_length] action history
            sample_next_states: [batch_size] next state indices
            sample_rewards: [batch_size] rewards
            masks: [batch_size, history_length] validity mask
            ground_truth_timesteps: [batch_size, history_length] ground truth timesteps
            mode: Evaluation mode
        
        Returns:
            Target Q-values
        """
        if mode is None:
            mode = self.q_format
        
        if mode == fqi.MULTIPLE_HEADS:
            # Use loaded optimal Q* for targets if available and requested
            if self.use_optimal_q_targets and self.optimal_q is not None:
                # Compute V*(s') using optimal Q
                v_next = q_iteration.logsumexp(
                    self.optimal_q[sample_next_states], 
                    alpha=self.ent_wt
                )
                target_q = ptu.tensor(sample_rewards + self.discount * v_next)
                
                # Expand to [batch_size, num_actions] for all actions
                target_q = target_q.unsqueeze(1).expand(-1, self.env.num_actions)
            else:
                # Use standard backup from current Q
                sample_histories_tensor = ptu.tensor(sample_histories, dtype=torch.int64)
                target_q = torch.index_select(
                    self.all_target_q, 0, sample_histories_tensor[:, -1]
                )  # [batch_size, num_actions]
        else:
            # FLAT mode
            if self.use_optimal_q_targets and self.optimal_q is not None:
                v_next = q_iteration.logsumexp(
                    self.optimal_q[sample_next_states],
                    alpha=self.ent_wt
                )
            else:
                v_next = q_iteration.logsumexp(
                    self.current_q[sample_next_states],
                    alpha=self.ent_wt
                )
            target_q = ptu.tensor(sample_rewards + self.discount * v_next)
        
        return target_q
    
    def project(self, network=None, optimizer=None, sampler=None):
        """
        Project onto Q-function space using gradient descent.
        Modified to handle history-conditioned inputs.
        """
        if network is None:
            network = self.network
        if optimizer is None:
            optimizer = self.qnet_optimizer
        if sampler is None:
            sampler = self.get_sample_states
        
        k = 0
        stopped_mode = None
        [stop_mode.reset() for stop_mode in self.stop_modes]
        
        for k in range(self.max_project_steps):
            # Sample with history, masks, and ground truth timesteps
            sample_data = sampler(itr=k)
            
            if len(sample_data) == 7:
                # Unpack: histories, actions, next_states, rewards, weights, masks, timesteps
                (sample_histories, sample_actions, sample_next_states, 
                 sample_rewards, weights, masks, ground_truth_timesteps) = sample_data
            else:
                # Fallback for compatibility
                sample_histories, sample_actions, sample_next_states, sample_rewards, weights = sample_data
                masks = None
                ground_truth_timesteps = None
            
            # Convert to tensors
            weights, = ptu.all_tensor([weights])
            
            # Evaluate target Q-values
            target_q = self.evaluate_target(
                sample_histories, sample_actions, sample_next_states, sample_rewards,
                masks=masks, ground_truth_timesteps=ground_truth_timesteps
            ).detach()

            
            # Evaluate current Q-values
            q_values = self.evaluate_qvalues(
                sample_histories, sample_actions, masks=masks, network=network
            )
            
            # Compute loss based on format
            # TODO: need to handle different cases with the history conditioned network currently.
            if self.q_format == fqi.MULTIPLE_HEADS:
                if len(weights.shape) == 2:
                    critic_loss = torch.mean(weights * (q_values - target_q) ** 2)
                else:
                    critic_loss = torch.mean(weights * torch.mean((q_values - target_q) ** 2, dim=1))
            else:
                critic_loss = torch.mean(weights * (q_values - target_q) ** 2)
            
            # Backprop
            network.zero_grad()
            critic_loss.backward()
            optimizer.step()
            
            # Check stopping conditions
            stop_args = dict(
                critic_loss=ptu.to_numpy(critic_loss),
                q_network=network,
                all_target_q=self.all_target_q,
                fqi=self,
                discount=self.discount,
                ent_wt=self.ent_wt
            )
            [stop_mode.update(**stop_args) for stop_mode in self.stop_modes]
            
            if k >= self.min_project_steps:
                for stop_mode in self.stop_modes:
                    if stop_mode.check():
                        logger.log(f'Early stopping via {stop_mode}.')
                        stopped_mode = stop_mode
                        break
                if stopped_mode:
                    break
        
        return stopped_mode, critic_loss, k
    
    def eval_policy(self, render=False, n_rollouts=None):
        """
        Evaluate the current policy.
        
        For history-conditioned policies, we need to build up history during rollout.
        This is a simplified evaluation that uses the last state only.
        """
        # For now, use the standard Q-function evaluation
        # TODO: Implement proper history-based evaluation
        return utils.eval_policy_qfn(
            self.env, self.current_q, 
            n_rollout=n_rollouts or self.n_eval_trials,
            ent_wt=self.ent_wt,
            render=render
        )


