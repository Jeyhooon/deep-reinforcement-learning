import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_dims=(64, 64)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        "*** YOUR CODE HERE ***"
        self.h_layers = nn.ModuleList()
        # self.norm_layers = nn.ModuleList()
        last_dim = hidden_dims[0]

        self.input_layer = nn.Linear(state_size, last_dim)
        # self.input_norm_layer = nn.LayerNorm(last_dim)

        for h_dim in hidden_dims[1:]:
            self.h_layers.append(nn.Linear(last_dim, h_dim))
            # self.norm_layers.append(nn.LayerNorm(h_dim))
            last_dim = h_dim
        self.output_layer = nn.Linear(last_dim, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        # x = F.relu(self.input_norm_layer(self.input_layer(state)))
        # for norm_layer, h_layer in zip(self.norm_layers, self.h_layers):
        #     x = F.relu(norm_layer(h_layer(x)))

        x = F.relu(self.input_layer(state))
        for h_layer in self.h_layers:
            x = F.relu(h_layer(x))

        return self.output_layer(x)



# # solution model
# class QNetwork(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(QNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.layer_norm_1 = nn.LayerNorm(fc1_units)
#
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.layer_norm_2 = nn.LayerNorm(fc2_units)
#
#         self.fc3 = nn.Linear(fc2_units, action_size)
#
#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.layer_norm_1(self.fc1(state)))
#         x = F.relu(self.layer_norm_2(self.fc2(x)))
#         return self.fc3(x)