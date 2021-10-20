
import torch
import numpy as np
from itertools import count

import gc
import os
import os.path
import time
import tempfile
import random

# import wandb

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'


class MASAC(torch.nn.Module):
    def __init__(self,
                 replay_buffer_fn,
                 policy_model_fn,
                 policy_optimizer_fn,
                 value_model_fn,
                 value_optimizer_fn,
                 config):
        super(MASAC, self).__init__()

        self.replay_buffer_fn = replay_buffer_fn

        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_max_grad_norm = config["POLICY_MAX_GRAD_NORM"]
        self.policy_optimizer_lr = config["POLICY_LR"]

        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_max_grad_norm = config["Q_MAX_GRAD_NORM"]
        self.value_optimizer_lr = config["Q_LR"]

        self.n_warmup_batches = config["WARMUP_BATCHES"]
        self.update_target_every_steps = config["UPDATE_EVERY"]

        self.tau = config["TAU"]
        self.root_dir = config["ROOT_DIR"]
        self.num_agents = config["NUM_AGENTS"]

        self.config = config

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = states.shape[0]

        # policy loss
        alpha_loss = 0.0

        logpi_s_list = []
        current_actions_list = []
        for i in range(self.num_agents):
            current_actions, logpi_s, _ = self.policy_model.full_pass(states[:, i])
            current_actions_list.append(current_actions)
            logpi_s_list.append(logpi_s)

            target_alpha = (logpi_s + self.policy_model.target_entropy).detach()
            alpha_loss += -(self.policy_model.logalpha * target_alpha).mean()

        self.policy_model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy_model.alpha_optimizer.step()
        # alpha = self.policy_model.logalpha.exp()
        alpha = self.config["ALPHA"]

        # Q loss
        ap_list = []
        logpi_sp_list = []
        for i in range(self.num_agents):
            ap, logpi_sp, _ = self.policy_model.full_pass(next_states[:, i])
            ap_list.append(ap)
            logpi_sp_list.append(logpi_sp)
        ap = torch.cat(ap_list, dim=-1)

        # Updating Q-Agent_1
        q1_spap_a = self.critic_1_target_value_model_a(next_states.reshape((batch_size, -1)), ap)
        q1_spap_b = self.critic_1_target_value_model_b(next_states.reshape((batch_size, -1)), ap)
        q1_spap = torch.min(q1_spap_a, q1_spap_b) - alpha * logpi_sp_list[0]
        target_q1_sa = (rewards[:, 0] + self.gamma * q1_spap * (1 - is_terminals[:, 0])).detach()

        q1_sa_a = self.critic_1_online_value_model_a(states.reshape((batch_size, -1)), actions.reshape((batch_size, -1)))
        q1_sa_b = self.critic_1_online_value_model_b(states.reshape((batch_size, -1)), actions.reshape((batch_size, -1)))
        q1_a_loss = (q1_sa_a - target_q1_sa).pow(2).mul(0.5).mean()
        q1_b_loss = (q1_sa_b - target_q1_sa).pow(2).mul(0.5).mean()

        self.critic_1_value_optimizer_a.zero_grad()
        q1_a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1_online_value_model_a.parameters(),
                                       self.value_max_grad_norm)
        self.critic_1_value_optimizer_a.step()

        self.critic_1_value_optimizer_b.zero_grad()
        q1_b_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1_online_value_model_b.parameters(),
                                       self.value_max_grad_norm)
        self.critic_1_value_optimizer_b.step()

        # Updating Q-Agent_2
        q2_spap_a = self.critic_2_target_value_model_a(next_states.reshape((batch_size, -1)), ap)
        q2_spap_b = self.critic_2_target_value_model_b(next_states.reshape((batch_size, -1)), ap)
        q2_spap = torch.min(q2_spap_a, q2_spap_b) - alpha * logpi_sp_list[1]
        target_q2_sa = (rewards[:, 1] + self.gamma * q2_spap * (1 - is_terminals[:, 1])).detach()

        q2_sa_a = self.critic_2_online_value_model_a(states.reshape((batch_size, -1)), actions.reshape((batch_size, -1)))
        q2_sa_b = self.critic_2_online_value_model_b(states.reshape((batch_size, -1)), actions.reshape((batch_size, -1)))
        q2_a_loss = (q2_sa_a - target_q2_sa).pow(2).mul(0.5).mean()
        q2_b_loss = (q2_sa_b - target_q2_sa).pow(2).mul(0.5).mean()

        self.critic_2_value_optimizer_a.zero_grad()
        q2_a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2_online_value_model_a.parameters(),
                                       self.value_max_grad_norm)
        self.critic_2_value_optimizer_a.step()

        self.critic_2_value_optimizer_b.zero_grad()
        q2_b_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2_online_value_model_b.parameters(),
                                       self.value_max_grad_norm)
        self.critic_2_value_optimizer_b.step()

        # updating the Policy
        current_q1_sa_a = self.critic_1_online_value_model_a(states.reshape((batch_size, -1)), torch.cat([current_actions_list[0], current_actions_list[1].detach()], dim=-1))
        current_q1_sa_b = self.critic_1_online_value_model_b(states.reshape((batch_size, -1)), torch.cat([current_actions_list[0], current_actions_list[1].detach()], dim=-1))
        current_q1_sa = torch.min(current_q1_sa_a, current_q1_sa_b)
        policy_loss_1 = (alpha * logpi_s_list[0] - current_q1_sa).mean()

        current_q2_sa_a = self.critic_2_online_value_model_a(states.reshape((batch_size, -1)), torch.cat([current_actions_list[0].detach(), current_actions_list[1]], dim=-1))
        current_q2_sa_b = self.critic_2_online_value_model_b(states.reshape((batch_size, -1)), torch.cat([current_actions_list[0].detach(), current_actions_list[1]], dim=-1))
        current_q2_sa = torch.min(current_q2_sa_a, current_q2_sa_b)
        policy_loss_2 = (alpha * logpi_s_list[1] - current_q2_sa).mean()

        policy_loss = policy_loss_1 + policy_loss_2
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(),
                                       self.policy_max_grad_norm)
        self.policy_optimizer.step()

        log_dict = {"critic_1_a_loss": q1_a_loss.item(), "critic_1_b_loss": q1_b_loss.item(),
                    "critic_2_a_loss": q2_a_loss.item(), "critic_2_b_loss": q2_b_loss.item(),
                    "logpi_s_1": logpi_s_list[0], "logpi_s_2": logpi_s_list[1], "logpi_sp_1": logpi_sp_list[0], "logpi_sp_2": logpi_sp_list[1],
                    "policy_loss": policy_loss.item(), "alpha_loss": alpha_loss.item(), "alpha": alpha,
                    "agent1_acts_dim1": current_actions_list[0][:, 0], "agent1_acts_dim2": current_actions_list[0][:, 1],
                    "agent2_acts_dim1": current_actions_list[1][:, 0], "agent2_acts_dim2": current_actions_list[1][:, 1]}
        # wandb.log(log_dict)

    def interaction_step(self, states, env):
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
        if len(self.replay_buffer) < min_samples:
            actions = [self.policy_model.select_random_action() for _ in range(self.num_agents)]
        else:
            actions = [self.policy_model.select_action(states[i]) for i in range(self.num_agents)]

        env_info = env.step(np.concatenate(actions, axis=-1))[self.brain_name]
        new_states, rewards, is_terminal = env_info.vector_observations, env_info.rewards, env_info.local_done
        experience = (states, actions, rewards, new_states, [float(is_terminal[i]) for i in range(self.num_agents)])

        self.replay_buffer.store(experience)
        self.episode_timestep[-1] += 1
        for i in range(self.num_agents):
            self.episode_rewards[i][-1] += rewards[i]

        return new_states, is_terminal

    def update_value_networks(self, tau=None):
        tau = self.tau if tau is None else tau

        for target, online in zip(self.critic_1_target_value_model_a.parameters(),
                                  self.critic_1_online_value_model_a.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.critic_1_target_value_model_b.parameters(),
                                  self.critic_1_online_value_model_b.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)


        for target, online in zip(self.critic_2_target_value_model_a.parameters(),
                                  self.critic_2_online_value_model_a.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.critic_2_target_value_model_b.parameters(),
                                  self.critic_2_online_value_model_b.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def setup(self, nS, nA, acts_bounds):

        self.critic_1_target_value_model_a = self.value_model_fn(nS*2, nA*2)
        self.critic_1_online_value_model_a = self.value_model_fn(nS*2, nA*2)
        self.critic_1_target_value_model_b = self.value_model_fn(nS*2, nA*2)
        self.critic_1_online_value_model_b = self.value_model_fn(nS*2, nA*2)

        self.critic_2_target_value_model_a = self.value_model_fn(nS*2, nA*2)
        self.critic_2_online_value_model_a = self.value_model_fn(nS*2, nA*2)
        self.critic_2_target_value_model_b = self.value_model_fn(nS*2, nA*2)
        self.critic_2_online_value_model_b = self.value_model_fn(nS*2, nA*2)

        self.update_value_networks()

        self.policy_model = self.policy_model_fn(nS, acts_bounds)

        self.critic_1_value_optimizer_a = self.value_optimizer_fn(self.critic_1_online_value_model_a,
                                                                  self.value_optimizer_lr)
        self.critic_1_value_optimizer_b = self.value_optimizer_fn(self.critic_1_online_value_model_b,
                                                                  self.value_optimizer_lr)

        self.critic_2_value_optimizer_a = self.value_optimizer_fn(self.critic_2_online_value_model_a,
                                                                  self.value_optimizer_lr)
        self.critic_2_value_optimizer_b = self.value_optimizer_fn(self.critic_2_online_value_model_b,
                                                                  self.value_optimizer_lr)

        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model,
                                                         self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()

    def train(self, env, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = tempfile.mkdtemp()
        self.seed = seed
        self.gamma = gamma

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.brain_name = env.brain_names[0]  # get the default brain
        self.brain = env.brains[self.brain_name]
        env_info = env.reset(train_mode=True)[self.brain_name]

        nS, nA = env_info.vector_observations.shape[1], self.brain.vector_action_space_size
        action_bounds = [-1 for _ in range(nA)], [1 for _ in range(nA)]
        self.setup(nS, nA, action_bounds)

        self.episode_timestep = []
        self.episode_rewards = [[], []]
        self.episode_seconds = []
        self.evaluation_scores = [[], []]
        self.episode_exploration = []
        self.episode_eval_max_reward = []

        result = {}
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            env_info = env.reset(train_mode=True)[self.brain_name]
            states, is_terminal = env_info.vector_observations, [False, False]

            [self.episode_rewards[i].append(0.0) for i in range(self.num_agents)]
            self.episode_timestep.append(0.0)
            self.episode_eval_max_reward.append(0.0)

            for _ in count():
                states, is_terminal = self.interaction_step(states, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.replay_buffer.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_value_networks()

                if is_terminal:
                    gc.collect()
                    break

            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.policy_model, env)
            self.save_checkpoint(episode - 1, self.policy_model)

            total_step = int(np.sum(self.episode_timestep))
            [self.evaluation_scores[i].append(evaluation_score[i]) for i in range(self.num_agents)]
            self.episode_eval_max_reward.append(np.max(evaluation_score))
            log_dict = {"eval_reward_agent_1": self.evaluation_scores[0][-1],
                        "eval_reward_agent_2": self.evaluation_scores[1][-1],
                        "eval_max_reward_mean_100": np.mean(self.episode_eval_max_reward[-100:]),
                        "eval_max_reward_std_100": np.std(self.episode_eval_max_reward[-100:]),
                        "eval_episode": episode}
            # wandb.log(log_dict)

            mean_10_reward = [np.mean(self.episode_rewards[i][-10:]) for i in range(self.num_agents)]
            std_10_reward = [np.std(self.episode_rewards[i][-10:]) for i in range(self.num_agents)]
            mean_100_reward = [np.mean(self.episode_rewards[i][-100:]) for i in range(self.num_agents)]
            std_100_reward = [np.std(self.episode_rewards[i][-100:]) for i in range(self.num_agents)]
            mean_100_eval_score = [np.mean(self.evaluation_scores[i][-100:]) for i in range(self.num_agents)]
            std_100_eval_score = [np.std(self.evaluation_scores[i][-100:]) for i in range(self.num_agents)]

            wallclock_elapsed = time.time() - training_start
            result.update({episode - 1:
                               (total_step, [self.episode_rewards[i][episode - 1] for i in range(self.num_agents)],\
                                mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed)})

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = any(np.array(mean_100_reward) >= goal_mean_100_reward)
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = f'elapsed {elapsed_str}, episode {episode - 1}, steps {total_step}, '
            debug_message += f'ave_r 10 {np.array(mean_10_reward)}\u00B1{np.array(std_10_reward)}, '
            debug_message += f'ave_r 100 {np.array(mean_100_reward)}\u00B1{np.array(std_100_reward)}, '
            debug_message += f'ave_eval 100 {np.array(mean_100_eval_score)}\u00B1{np.array(std_100_eval_score)}'
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward of +30 over last 100 episodes \u2713')
                break

        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print(f'Final evaluation score {final_eval_score}\u00B1{score_std} in {training_time}s training time,'
              f' {wallclock_time}s wall-clock time.\n')

        self.save_checkpoint('final', self.policy_model)
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = [[], []]
        for _ in range(n_episodes):
            env_info = eval_env.reset(train_mode=True)[self.brain_name]
            s, d = env_info.vector_observations, [False, False]
            [rs[i].append(0) for i in range(self.num_agents)]
            for _ in count():
                a = np.concatenate([eval_policy_model.select_greedy_action(s[i]) for i in range(self.num_agents)], axis=-1)
                env_info = eval_env.step(a)[self.brain_name]
                s, r, d = env_info.vector_observations, env_info.rewards, env_info.local_done
                for i in range(self.num_agents): rs[i][-1] += r[i]
                if any(d): break
        return [np.mean(rs[i]) for i in range(self.num_agents)], [np.std(rs[i]) for i in range(self.num_agents)]

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(),
                   os.path.join(self.root_dir if episode_idx == 'final' else self.checkpoint_dir,
                                'model_{}.pth'.format(episode_idx)))
