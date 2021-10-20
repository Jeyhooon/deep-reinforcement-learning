
import argparse
import numpy as np
import torch
from torch import optim
import pathlib
import os
import matplotlib.pyplot as plt
from datetime import date
# import wandb

from unityagents import UnityEnvironment
from scripts.sac_agent import MASAC
from scripts import utils_sac as utils

os.chdir(pathlib.Path(__file__).parent.absolute())
RESULTS_DIR = os.path.join('results')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {
    "NUM_AGENTS": 2,
    "ROOT_DIR": "results",                  # directory to save the results
    "BUFFER_SIZE": int(1e6),                # replay buffer size
    "BATCH_SIZE": 256,                      # mini-batch size
    "WARMUP_BATCHES": 10,                   # number of initial batches to fill the buffer with
    "TAU": 5e-3,                            # for soft update of target parameters
    "ALPHA": 0.1,
    "UPDATE_EVERY": 4,                      # how often to update the network
    "SEED": [64],                            # list of the seed to do randomize each training
    "Q_NET_Hidden_Dims": (1024, 1024),          # Size of the hidden layer in Q-Net
    "Q_LR": 3e-4,                           # Q-Net learning rate
    "Q_MAX_GRAD_NORM": float('inf'),        # to clip gradients of Q-Net
    "POLICY_NET_Hidden_Dims": (1024, 1024),     # Size of the hidden layer in Policy-Net
    "POLICY_LR": 3e-4,                      # Policy-Net learning rate
    "POLICY_MAX_GRAD_NORM": float(10.0),   # to clip gradients of the Policy-Net
    "WEIGHT_DECAY": float(3e-5),

    "ENV_SETTINGS": {
            'ENV_NAME': 'data/Tennis_Linux/Tennis.x86_64',
            'GAMMA': 0.99,
            'MAX_MINUTES': 300,
            'MAX_EPISODES': int(15e3),
            'GOAL_MEAN_100_REWARD': 0.5
        }
}


def create_agent(config):

    policy_model_fn = lambda nS, bounds: utils.GaussianPolicyNet(nS, bounds, hidden_dims=config["POLICY_NET_Hidden_Dims"])
    policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr, weight_decay=config["WEIGHT_DECAY"])

    value_model_fn = lambda nS, nA: utils.QNet(nS, nA, hidden_dims=config["Q_NET_Hidden_Dims"])
    value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr, weight_decay=config["WEIGHT_DECAY"])

    replay_buffer_fn = lambda: utils.ReplayBuffer(buffer_size=config["BUFFER_SIZE"], batch_size=config["BATCH_SIZE"])

    return MASAC(replay_buffer_fn,
                 policy_model_fn,
                 policy_optimizer_fn,
                 value_model_fn,
                 value_optimizer_fn,
                 config)


def process_results(results, root_dir: str):
    '''
        Extracts Relevent information, Plots and Saves the Results
    '''

    max_total_steps, max_episode_reward, max_100_reward, max_100_score, \
    max_train_time, max_wall_time = np.max(results, axis=0).T

    min_total_steps, min_episode_reward, min_100_reward, min_100_score, \
    min_train_time, min_wall_time = np.min(results, axis=0).T

    mean_total_steps, mean_episode_reward, mean_100_reward, mean_100_score, \
    mean_train_time, mean_wall_time = np.mean(results, axis=0).T

    x = np.arange(len(mean_100_score))

    stats_dict = {
        'x': x,

        'max_100_reward': max_100_reward,
        'min_100_reward': min_100_reward,
        'mean_100_reward': mean_100_reward,

        'max_100_score': max_100_score,
        'min_100_score': min_100_score,
        'mean_100_score': mean_100_score,

        'max_total_steps': max_total_steps,
        'min_total_steps': min_total_steps,
        'mean_total_steps': mean_total_steps,

        'max_train_time': max_train_time,
        'min_train_time': min_train_time,
        'mean_train_time': mean_train_time,

        'max_wall_time': max_wall_time,
        'min_wall_time': min_wall_time,
        'mean_wall_time': mean_wall_time
    }

    data_path = os.path.join(root_dir, 'stats_dict.pth')
    torch.save(stats_dict, data_path)
    print(f"Processed Data Saved to: {data_path}")

    # Plot the Learning Curve
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharey=False, sharex=True)

    axs[0].plot(max_100_reward, 'g', linewidth=1)
    axs[0].plot(min_100_reward, 'g', linewidth=1)
    axs[0].plot(mean_100_reward, 'g:', label='SAC', linewidth=2)
    axs[0].fill_between(
        x, min_100_reward, max_100_reward, facecolor='g', alpha=0.3)

    axs[1].plot(max_episode_reward, 'g', linewidth=1)
    axs[1].plot(min_episode_reward, 'g', linewidth=1)
    axs[1].plot(mean_episode_reward, 'g:', label='SAC', linewidth=2)
    axs[1].fill_between(
        x, min_episode_reward, max_episode_reward, facecolor='g', alpha=0.3)

    axs[0].set_title('Moving Avg. Last_100_Episode_Reward (Training)')
    axs[1].set_title('Mean Episode Rewards (Training)')
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')

    lc_path = os.path.join(root_dir, 'learning_curve.png')
    plt.savefig(lc_path)
    print(f"Learning-Curve Saved to: {lc_path}")
    plt.show()


    # Plot training time stats
    fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharey=False, sharex=True)

    axs[0].plot(max_total_steps, 'g', linewidth=1)
    axs[0].plot(min_total_steps, 'g', linewidth=1)
    axs[0].plot(mean_total_steps, 'g:', label='SAC', linewidth=2)
    axs[0].fill_between(x, min_total_steps, max_total_steps, facecolor='g', alpha=0.3)

    axs[1].plot(max_train_time, 'g', linewidth=1)
    axs[1].plot(min_train_time, 'g', linewidth=1)
    axs[1].plot(mean_train_time, 'g:', label='SAC', linewidth=2)
    axs[1].fill_between(x, min_train_time, max_train_time, facecolor='g', alpha=0.3)

    axs[2].plot(max_wall_time, 'g', linewidth=1)
    axs[2].plot(min_wall_time, 'g', linewidth=1)
    axs[2].plot(mean_wall_time, 'g:', label='SAC', linewidth=2)
    axs[2].fill_between(x, min_wall_time, max_wall_time, facecolor='g', alpha=0.3)

    axs[0].set_title('Total Steps')
    axs[1].set_title('Training Time')
    axs[2].set_title('Wall-clock Time')
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')

    tc_path = os.path.join(root_dir, 'training_time_stats.png')
    plt.savefig(tc_path)
    print(f"Training-Time-Stats Saved to: {tc_path}")
    plt.show()


def train(env):

    # Creating the required directories
    current_date = str(date.today()).replace('-', '_')
    current_time = utils.get_date_time_now()
    config["ROOT_DIR"] = os.path.join(config["ROOT_DIR"], current_date, current_time)
    os.makedirs(config["ROOT_DIR"], exist_ok=True)

    exp_results = []
    best_agent, best_eval_score = None, float('-inf')
    for seed in config["SEED"]:

        _, gamma, max_minutes, max_episodes, goal_mean_100_reward = config["ENV_SETTINGS"].values()

        agent = create_agent(config)
        result, final_eval_score, training_time, wallclock_time = agent.train(env,
                                                                              seed,
                                                                              gamma,
                                                                              max_minutes,
                                                                              max_episodes,
                                                                              goal_mean_100_reward)

        exp_results.append(result)
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent = agent

    process_results(exp_results, config["ROOT_DIR"])
    return best_agent


if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    # Parsing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=bool, default=True, help='Train otherwise Test/Eval')
    parser.add_argument('--load_dir', type=str, default=None, help='Directory to load the model from')
    args = parser.parse_args()

    # wandb_id = wandb.util.generate_id()
    date_time_now = utils.get_date_time_now()
    # wandb.init(project="MASAC_Tennis", id=wandb_id, config=config)
    # wandb.run.name = date_time_now + "__" + wandb_id
    # wandb.run.save()

    env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = config["ENV_SETTINGS"].values()
    env = UnityEnvironment(file_name=env_name, seed=config["SEED"][0])

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    # number of agents in the environment
    print('Number of Agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Action Size:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('State Size:', state_size)
    print('States look like:', state, '\n')

    # create and setup the agents
    num_agents = config["NUM_AGENTS"]
    agent = create_agent(config)
    action_bounds = [-1 for _ in range(action_size)], [1 for _ in range(action_size)]
    agent.setup(state_size, action_size, action_bounds)
    # wandb.watch(agent, log='all')

    # watch an untrained agent
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations

    # initialize the scores
    agent_1_score = 0
    agent_2_score = 0

    for _ in range(50):
        actions = np.concatenate([agent.policy_model.select_action(states[i]) for i in range(num_agents)])  # select an action
        env_info = env.step(actions)[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get the next state
        rewards = env_info.rewards  # get the reward
        dones = env_info.local_done  # see if episode has finished
        agent_1_score += rewards[0]  # update the score
        agent_2_score += rewards[1]  # update the score
        states = next_states  # roll over the state to next time step
        if any(dones):  # exit loop if episode finished
            break
    print(f"UnTrained Agents Score: agent_1: {agent_1_score},  agent_2: {agent_2_score}")

    if args.is_training:
        agent = train(env)
        args.load_dir = config["ROOT_DIR"]

    # load the weights from the file
    assert args.load_dir is not None
    trained_policy = utils.load_checkpoint(model=agent.policy_model, path=args.load_dir)

    # watch the trained agent
    score = 0  # initialize the score
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    while True:
        action = trained_policy.select_action(state)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break
    print("Trained Agent's Score: {}".format(score))

    print("Experiment Finished! ... Closing the Environment ...")
    env.close()
