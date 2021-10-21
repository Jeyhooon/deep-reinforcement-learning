
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from datetime import datetime
import subprocess
import sys


def get_date_time_now():
    return str(datetime.now().replace(microsecond=0)).replace(':', '_').replace(' ', '_')


def install_pkg(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_req_pkgs():
    subprocess.check_call([sys.executable, "-q", "pip", "install", '../../python'])


def load_checkpoint(model, path: str):
    model.load_state_dict(torch.load(path))
    return model


class ReplayBuffer:
    def __init__(self, buffer_size=10000, batch_size=64):

        self.state_mem = np.empty(shape=(buffer_size), dtype=np.ndarray)
        self.action_mem = np.empty(shape=(buffer_size), dtype=np.ndarray)
        self.reward_mem = np.empty(shape=(buffer_size), dtype=np.ndarray)
        self.next_state_mem = np.empty(shape=(buffer_size), dtype=np.ndarray)
        self.is_done_mem = np.empty(shape=(buffer_size), dtype=np.ndarray)

        self.max_size = buffer_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        state, action, reward, next_state, is_done = sample
        self.state_mem[self._idx] = state
        self.action_mem[self._idx] = action
        self.reward_mem[self._idx] = reward
        self.next_state_mem[self._idx] = next_state
        self.is_done_mem[self._idx] = is_done

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)
        experiences = np.vstack(self.state_mem[idxs]), \
                      np.vstack(self.action_mem[idxs]), \
                      np.vstack(self.reward_mem[idxs]), \
                      np.vstack(self.next_state_mem[idxs]), \
                      np.vstack(self.is_done_mem[idxs])
        return experiences

    def __len__(self):
        return self.size


class GaussianPolicyNet(nn.Module):
    def __init__(self,
                 input_dim,
                 action_bounds,
                 log_std_min=-20,
                 log_std_max=2,
                 hidden_dims=(32, 32),
                 activation_fc=torch.relu,
                 entropy_lr=0.001):
        super(GaussianPolicyNet, self).__init__()
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.input_layer = nn.Linear(input_dim,
                                     hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(
                hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer_mean = nn.Linear(hidden_dims[-1], len(self.env_max))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1], len(self.env_max))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min,
                                    device=self.device,
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device,
                                    dtype=torch.float32)

        self.action_shape = self.env_max.cpu().numpy().shape

        self.nn_min = torch.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = torch.tanh(torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min

        self.target_entropy = float(-np.prod(self.env_max.shape))
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.logalpha], lr=entropy_lr)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x_mean = self.output_layer_mean(x)
        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std,
                                self.log_std_min,
                                self.log_std_max)
        return x_mean, x_log_std

    def full_pass(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)

        pi_s = Normal(mean, log_std.exp())
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale_fn(tanh_action)

        log_prob = pi_s.log_prob(pre_tanh_action) - \
                   torch.log((1 - tanh_action.pow(2)).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, self.rescale_fn(torch.tanh(mean))

    def select_random_action(self):
        return np.random.uniform(low=self.env_min.cpu().numpy(),
                                 high=self.env_max.cpu().numpy()).reshape(self.action_shape)

    def select_greedy_action(self, state):
        return self.rescale_fn(torch.tanh(self.forward(state)[0])).detach().cpu().numpy().reshape(self.action_shape)

    def select_action(self, state):
        mean, log_std = self.forward(state)
        action = self.rescale_fn(torch.tanh(Normal(mean, log_std.exp()).sample()))
        return action.detach().cpu().numpy().reshape(self.action_shape)


class QNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(QNet, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim + output_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u,
                             device=self.device,
                             dtype=torch.float32)
            u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action) -> torch.Tensor:
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_layer(torch.cat((x, u), dim=1)))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x

    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals
