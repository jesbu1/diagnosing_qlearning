import numpy as np
import torch

from debugq import pytorch_util as ptu

class TabularNetwork(torch.nn.Module):
  def __init__(self, env):
    super(TabularNetwork, self).__init__()
    self.num_states = env.num_states
    self.network = torch.nn.Sequential(
        torch.nn.Linear(self.num_states, env.num_actions)
    )

  def forward(self, states):
    onehot = ptu.one_hot(states, self.num_states)
    return self.network(onehot)

  def reset_weights(self):
    for layer in self.network:
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  

def stack_observations(env):
    obs = []
    for s in range(env.num_states):
        obs.append(env.observation(s))
    return np.stack(obs)


class LinearNetwork(torch.nn.Module):
  def __init__(self, env):
    super(LinearNetwork, self).__init__()
    self.all_observations = ptu.tensor(stack_observations(env))
    self.dim_input = env.observation_space.shape[-1]
    self.network = torch.nn.Sequential(
        torch.nn.Linear(self.dim_input, env.num_actions)
    )

  def forward(self, states):
    observations = torch.index_select(self.all_observations, 0, states) 
    return self.network(observations)

  def reset_weights(self):
    for layer in self.network:
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()


class FCNetwork(torch.nn.Module):
  def __init__(self, env, layers=[20,20]):
    super(FCNetwork, self).__init__()
    self.all_observations = ptu.tensor(stack_observations(env))
    dim_input = env.observation_space.shape
    dim_output = env.num_actions
    net_layers = []

    dim = dim_input[-1]
    for i, layer_size in enumerate(layers):
      net_layers.append(torch.nn.Linear(dim, layer_size))
      net_layers.append(torch.nn.ReLU())
      dim = layer_size
    net_layers.append(torch.nn.Linear(dim, dim_output))
    self.layers = net_layers
    self.network = torch.nn.Sequential(*net_layers)

  def forward(self, states):
    observations = torch.index_select(self.all_observations, 0, states) 
    return self.network(observations)

  def reset_weights(self):
    for layer in self.network:
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()


class HistoryConditionedQNetwork(torch.nn.Module):
  """
  Q-network that conditions on a history of observations.
  
  Args:
    env: Environment
    history_encoder: Type of encoder ('lstm', 'gru', 'transformer')
    hidden_dim: Hidden dimension for the encoder
    num_layers: Number of recurrent layers
  """
  def __init__(self, env, history_encoder='lstm', hidden_dim=64, num_layers=1):
    super(HistoryConditionedQNetwork, self).__init__()
    self.all_observations = ptu.tensor(stack_observations(env))
    self.num_states = env.num_states
    self.obs_dim = env.observation_space.shape[-1]
    self.num_actions = env.num_actions
    self.hidden_dim = hidden_dim
    self.history_encoder_type = history_encoder
    
    # Encoder for history
    if history_encoder == 'lstm':
      self.encoder = torch.nn.LSTM(
        input_size=self.obs_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True
      )
    elif history_encoder == 'gru':
      self.encoder = torch.nn.GRU(
        input_size=self.obs_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True
      )
    else:
      raise ValueError(f"Unknown history encoder: {history_encoder}")
    
    # Output head for Q-values
    self.q_head = torch.nn.Sequential(
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, self.num_actions)
    )
  
  def forward(self, history_states, mask=None):
    """
    Forward pass through the network.
    
    Args:
      history_states: Tensor of shape [batch, history_length] containing state indices
                     or [batch, history_length, obs_dim] containing observations
      mask: Optional tensor of shape [batch, history_length] indicating valid timesteps
    
    Returns:
      Q-values of shape [batch, num_actions]
    """
    # Convert state indices to observations if needed
    if history_states.dim() == 2:
      # Input is [batch, history_length] state indices
      batch_size, history_length = history_states.shape
      # Flatten to [batch * history_length]
      flat_states = history_states.reshape(-1)
      # Look up observations: [batch * history_length, obs_dim]
      flat_obs = torch.index_select(self.all_observations, 0, flat_states)
      # Reshape to [batch, history_length, obs_dim]
      observations = flat_obs.reshape(batch_size, history_length, self.obs_dim)
    else:
      # Input is already observations [batch, history_length, obs_dim]
      observations = history_states
    
    # Encode history
    if self.history_encoder_type in ['lstm', 'gru']:
      # encoder output: [batch, seq_len, hidden_dim]
      # hidden: [num_layers, batch, hidden_dim]
      encoder_output, hidden = self.encoder(observations)
      
      if mask is not None:
        # Use mask to select the last valid timestep for each sequence
        # mask: [batch, history_length]
        # Find the index of the last valid timestep
        lengths = mask.sum(dim=1).long() - 1  # [batch]
        lengths = torch.clamp(lengths, min=0)
        # Gather the output at the last valid position
        batch_indices = torch.arange(observations.size(0), device=observations.device)
        final_hidden = encoder_output[batch_indices, lengths]  # [batch, hidden_dim]
      else:
        # Use the last timestep
        final_hidden = encoder_output[:, -1, :]  # [batch, hidden_dim]
    else:
      raise NotImplementedError(f"Encoder {self.history_encoder_type} not implemented")
    
    # Compute Q-values
    q_values = self.q_head(final_hidden)  # [batch, num_actions]
    
    return q_values
  
  def reset_weights(self):
    """Reset all network parameters"""
    for module in [self.encoder, self.q_head]:
      if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
      else:
        for layer in module:
          if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()