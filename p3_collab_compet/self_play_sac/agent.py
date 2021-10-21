
import torch
import numpy as np
from itertools import count

import gc
import os
import os.path
import time
import tempfile
import random
import mlflow


LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'


class SACAgent:
    def __init__(self,
                 replay_buffer_fn,
                 policy_model_fn,
                 policy_optimizer_fn,
                 value_model_fn,
                 value_optimizer_fn,
                 config):

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

    def setup(self, nS, nA, acts_bounds):

        self.target_value_model_a = self.value_model_fn(nS, nA)
        self.online_value_model_a = self.value_model_fn(nS, nA)

        self.target_value_model_b = self.value_model_fn(nS, nA)
        self.online_value_model_b = self.value_model_fn(nS, nA)

        self.update_value_networks()

        self.policy_model = self.policy_model_fn(nS, acts_bounds)

        self.value_optimizer_a = self.value_optimizer_fn(self.online_value_model_a,
                                                         self.value_optimizer_lr)
        self.value_optimizer_b = self.value_optimizer_fn(self.online_value_model_b,
                                                         self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model,
                                                         self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = is_terminals.shape[0]

        # policy loss
        current_actions, logpi_s, _ = self.policy_model.full_pass(states)

        target_alpha = (logpi_s + self.policy_model.target_entropy).detach()
        alpha_loss = -(self.policy_model.logalpha * target_alpha).mean()

        self.policy_model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.policy_model.alpha_optimizer.step()
        alpha = self.policy_model.logalpha.exp()

        # Q loss
        ap, logpi_sp, _ = self.policy_model.full_pass(next_states)
        q_spap_a = self.target_value_model_a(next_states, ap)
        q_spap_b = self.target_value_model_b(next_states, ap)
        q_spap = torch.min(q_spap_a, q_spap_b) - alpha * logpi_sp
        target_q_sa = (rewards + self.gamma * q_spap * (1 - is_terminals)).detach()

        q_sa_a = self.online_value_model_a(states, actions)
        q_sa_b = self.online_value_model_b(states, actions)
        qa_loss = (q_sa_a - target_q_sa).pow(2).mul(0.5).mean()
        qb_loss = (q_sa_b - target_q_sa).pow(2).mul(0.5).mean()

        self.value_optimizer_a.zero_grad()
        qa_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model_a.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer_a.step()

        self.value_optimizer_b.zero_grad()
        qb_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model_b.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer_b.step()

        current_q_sa_a = self.online_value_model_a(states, current_actions)
        current_q_sa_b = self.online_value_model_b(states, current_actions)
        current_q_sa = torch.min(current_q_sa_a, current_q_sa_b)
        policy_loss = (alpha * logpi_s - current_q_sa).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(),
                                       self.policy_max_grad_norm)
        self.policy_optimizer.step()

        log_dict = {"critic_a_loss": qa_loss.item(), "critic_b_loss": qb_loss.item(),
                    "logpi_s": logpi_s, "logpi_sp": logpi_sp,
                    "policy_loss": policy_loss.item(), "alpha_loss": alpha_loss.item(), "alpha": alpha}
        for k in log_dict.keys():
            if type(log_dict[k]) == float:
                mlflow.log_metric(k, log_dict[k], self.episode)

    def interaction_step(self, states, env):
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches

        if len(self.replay_buffer) < min_samples:
            actions = np.array([self.policy_model.select_random_action() for _ in range(self.num_agents)])
        else:
            actions = np.array([self.policy_model.select_action(states[i]) for i in range(self.num_agents)])

        env_info = env.step(actions)[self.brain_name]
        new_states, rewards, is_terminal = env_info.vector_observations, np.array(env_info.rewards), np.array(env_info.local_done, dtype=np.float32)

        [self.replay_buffer.store((states[i], actions[i], rewards[i], new_states[i], is_terminal[i])) for i in range(self.num_agents)]

        self.episode_reward[-1] += rewards
        self.episode_timestep[-1] += 1
        return new_states, is_terminal

    def update_value_networks(self, tau=None):
        tau = self.tau if tau is None else tau

        for target, online in zip(self.target_value_model_a.parameters(),
                                  self.online_value_model_a.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_value_model_b.parameters(),
                                  self.online_value_model_b.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

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
        self.num_agents = len(env_info.agents)

        nS, nA = self.brain.vector_observation_space_size*self.brain.num_stacked_vector_observations, self.brain.vector_action_space_size
        action_bounds = [-1 for _ in range(nA)], [1 for _ in range(nA)]
        self.setup(nS, nA, action_bounds)

        self.episode_timestep = []
        self.episode_seconds = []
        self.episode_reward = np.zeros((1, self.num_agents))
        self.evaluation_scores = []

        result = np.empty((max_episodes, 6))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            self.episode = episode

            env_info = env.reset(train_mode=True)[self.brain_name]
            state, is_terminal, rewards = env_info.vector_observations, np.array(env_info.local_done, dtype=np.float32), np.array(env_info.rewards)

            if episode != 0: self.episode_reward = np.concatenate([self.episode_reward, np.zeros((1, self.num_agents))], axis=0)
            self.episode_timestep.append(0.0)

            for _ in count():
                state, is_terminal = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_value_model_a.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_value_networks()

                if np.any(is_terminal):
                    gc.collect()
                    break

            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.policy_model, env)
            self.save_checkpoint(episode - 1, self.policy_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(np.max(self.episode_reward[-10:, :], axis=-1))
            std_10_reward = np.std(np.max(self.episode_reward[-10:, :], axis=-1))
            mean_100_reward = np.mean(np.max(self.episode_reward[-100:, :], axis=-1))
            std_100_reward = np.std(np.max(self.episode_reward[-100:, :], axis=-1))
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])

            log_dict = {"mean_10_reward": mean_10_reward,
                        "std_10_reward": std_10_reward,
                        "mean_100_reward": mean_100_reward,
                        "std_100_reward": std_100_reward,
                        "mean_100_eval_score": mean_100_eval_score,
                        "std_100_eval_score": std_100_eval_score}
            mlflow.log_metrics(log_dict, episode)

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = total_step, np.max(self.episode_reward[episode - 1]), mean_100_reward, \
                                  mean_100_eval_score, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_reward >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = f'elapsed {elapsed_str}, episode {episode - 1}, steps {total_step}, '
            debug_message += f'ave_r 10 {mean_10_reward}\u00B1{std_10_reward}, '
            debug_message += f'ave_r 100 {mean_100_reward}\u00B1{std_100_reward}, '
            debug_message += f'ave_eval 100 {mean_100_eval_score}\u00B1{std_100_eval_score}'

            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward of +0.5 over last 100 episodes \u2713')
                break

        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print(f'Final evaluation score {final_eval_score}\u00B1{score_std} in {training_time}s training time,'
              f' {wallclock_time}s wall-clock time.\n')

        self.save_checkpoint('final', self.policy_model)
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = np.zeros((1, self.num_agents))
        for i in range(n_episodes):
            env_info = eval_env.reset(train_mode=True)[self.brain_name]
            s, d = env_info.vector_observations, np.array(env_info.local_done, dtype=np.float32)
            if i != 0: rs = np.concatenate([rs, np.zeros((1, self.num_agents))], axis=0)
            for _ in count():
                a = np.array([eval_policy_model.select_greedy_action(s[i]) for i in range(self.num_agents)])
                env_info = eval_env.step(a)[self.brain_name]
                s, r, d = env_info.vector_observations, np.array(env_info.rewards), np.array(env_info.local_done, dtype=np.float32)
                rs[-1] += r
                if np.any(d): break
        return np.mean(np.max(rs, axis=-1)), np.std(np.max(rs, axis=-1))

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(),
                   os.path.join(self.root_dir if episode_idx == 'final' else self.checkpoint_dir,
                                'model_{}.pth'.format(episode_idx)))
