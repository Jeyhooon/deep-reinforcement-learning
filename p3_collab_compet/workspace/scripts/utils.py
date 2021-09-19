
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import wandb
from datetime import datetime


device='cpu'
dtype=torch.float32
num_agents = 2


def get_date_time_now():
    return str(datetime.now().replace(microsecond=0)).replace(':', '_').replace(' ', '_')


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        self.deque.append(transition)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return [np.stack(item) for item in transpose_list(samples)]

    def __len__(self):
        return len(self.deque)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        # self.input_norm = nn.BatchNorm1d(input_dim)
        # self.input_norm.weight.data.fill_(1)
        # self.input_norm.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = F.leaky_relu  # leaky_relu
        self.actor = actor
        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):

        if self.actor:
            # return a vector of the force
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))
            return F.tanh(h3)

        else:
            # critic network simply outputs a number
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))
            return h3


class DDPGAgent(nn.Module):
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor,
                 in_critic, hidden_in_critic, hidden_out_critic,
                 lr_actor=1e-3, lr_critic=1e-3, dtype=torch.float32):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)

        self.critic_1 = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_critic_1 = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.critic_2 = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_critic_2 = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic_1, self.critic_1)
        hard_update(self.target_critic_2, self.critic_2)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic, weight_decay=1.e-5)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, obs, noise=0.0):
        noise = + torch.tensor(noise * self.noise.noise(), dtype=dtype, device=device)
        wandb.log({'noise_added_to_acts': noise})
        action = self.actor(obs) + noise
        return action

    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs) + torch.tensor(noise * self.noise.noise(), dtype=dtype, device=device)
        return action


class SelfPlayDDPG(nn.Module):
    def __init__(self, discount_factor=0.95, tau=0.02, dtype=torch.float32):
        super(SelfPlayDDPG, self).__init__()
        self.dtype = dtype

        # shared actor network (each agent have its own local obs); we learn critic for agent_1, for calculating
        # the targets for agent_2 we simply use the negative value of agent_1 critic.
        # critic input = full_obs + all_agents_actions = 2*24 + 2+2 = 52
        self.ddpg_agent = DDPGAgent(24, 64, 64, 2, 52, 128, 128)
        self.num_agents = 2

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def _format(self, input_array):
        if type(input_array) != torch.Tensor:
            return torch.tensor(input_array, dtype=self.dtype, device=device)
        else:
            return input_array

    def get_actor(self):
        """get actors of all the agents in the MADDPG object"""
        return self.ddpg_agent.actor

    def get_target_actor(self):
        """get target_actors of all the agents in the MADDPG object"""
        return self.ddpg_agent.target_actor

    def act(self, local_obs: torch.Tensor, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        # local_obs.shape: [batch, num_agents, dim]
        return [self.ddpg_agent.act(local_obs[:, i], noise) for i in range(self.num_agents)]

    def target_act(self, local_obs: torch.Tensor, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        return [self.ddpg_agent.target_act(local_obs[:, i], noise) for i in range(self.num_agents)]

    def update(self, samples, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to obs[agent_number][parallel_agent]
        obs, action, reward, next_obs, done = samples   # ndarray[batch, num_agents, dim]

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = torch.cat(self.target_act(self._format(next_obs)), dim=-1)   # [batch, 2*acts_dim]
        target_critics_input = torch.cat([torch.cat([self._format(next_obs[:, i]) for i in range(self.num_agents)], dim=-1), target_actions], dim=-1)   # [batch, 2*obs_dim + 2*acts_dim]

        with torch.no_grad():
            q1_next = self.ddpg_agent.target_critic_1(target_critics_input)     # [batch, 1]
            q2_next = self.ddpg_agent.target_critic_2(target_critics_input)     # [batch, 1]

        # targets:
        y1 = self._format(reward[:, 0][..., None]) + self.discount_factor * q1_next * (1 - self._format(done[:, 0][..., None].astype(float)))
        y2 = self._format(reward[:, 1][..., None]) + self.discount_factor * q2_next * (1 - self._format(done[:, 1][..., None].astype(float)))

        critic_input = torch.cat([torch.cat([self._format(obs[:, i]) for i in range(self.num_agents)], dim=-1),
                                  torch.cat([self._format(action[:, i]) for i in range(self.num_agents)], dim=-1)], dim=-1).to(device)  # [batch, 2*obs_dim + 2*acts_dim]

        q1 = self.ddpg_agent.critic_1(critic_input)    # [batch, 1]
        q2 = self.ddpg_agent.critic_2(critic_input)    # [batch, 1]
        huber_loss = torch.nn.SmoothL1Loss()

        critic_1_loss = huber_loss(q1, y1.detach())
        self.ddpg_agent.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        self.ddpg_agent.critic_1_optimizer.step()

        critic_2_loss = huber_loss(q2, y2.detach())
        self.ddpg_agent.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.ddpg_agent.critic_2_optimizer.step()

        # update actor network using policy gradient
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        online_acts_list = [self.ddpg_agent.actor(self._format(obs[:, i])) for i in range(self.num_agents)]  # [ [batch, acts_dim], [batch, acts_dim] ]
        online_acts_1 = torch.cat((online_acts_list[0], online_acts_list[1].detach()), dim=-1)
        online_acts_2 = torch.cat((online_acts_list[0].detach(), online_acts_list[1]), dim=-1)

        q1_input = torch.cat([torch.cat([self._format(obs[:, i]) for i in range(self.num_agents)], dim=-1), online_acts_1], dim=-1)  # [batch, 2*obs_dim + 2*acts_dim]
        q2_input = torch.cat([torch.cat([self._format(obs[:, i]) for i in range(self.num_agents)], dim=-1), online_acts_2], dim=-1)  # [batch, 2*obs_dim + 2*acts_dim]

        # get the policy gradient
        actor_loss = -self.ddpg_agent.critic_1(q1_input).mean() + -self.ddpg_agent.critic_2(q2_input).mean()
        self.ddpg_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        self.ddpg_agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        c1l = critic_1_loss.cpu().detach().item()
        c2l = critic_2_loss.cpu().detach().item()
        logger.add_scalars('agent/losses',
                           {'critic_1 loss': c1l,
                            'critic_2 loss': c2l,
                            'actor_loss': al},
                           self.iter)
        log_dict = {'critic_1 loss': c1l,
                    'critic_2 loss': c2l,
                    'actor_loss': al}
        wandb.log(log_dict)
        wandb.log({'acts_agent_1': online_acts_list[0].cpu().detach()})
        wandb.log({'acts_agent_2': online_acts_list[1].cpu().detach()})

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        soft_update(self.ddpg_agent.target_actor, self.ddpg_agent.actor, self.tau)
        soft_update(self.ddpg_agent.target_critic_1, self.ddpg_agent.critic_1, self.tau)
        soft_update(self.ddpg_agent.target_critic_2, self.ddpg_agent.critic_2, self.tau)


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process

    """

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def transpose_list(mylist):
    return list(map(list, zip(*mylist)))


def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))
