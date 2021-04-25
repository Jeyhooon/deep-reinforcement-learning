
import argparse
import numpy as np
from collections import deque
import torch
from unityagents import UnityEnvironment
from scripts.agent import Agent
import pathlib
import os
import matplotlib.pyplot as plt

os.chdir(pathlib.Path(__file__).parent.absolute())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    "BUFFER_SIZE": int(1e5),        # replay buffer size
    "BATCH_SIZE": 64,               # minibatch size
    "GAMMA": 0.99,                  # discount factor
    "TAU": 1e-3,                    # for soft update of target parameters
    "LR": 5e-4,                     # learning rate
    "UPDATE_EVERY": 4,              # how often to update the network
    "SEED": 10,
    "N_EPISODS": 2000,              # Number of episodes to train
    "EPS_START": 1.0,               # Epsilon starting value
    "EPS_END": 0.01,                # Minimum epsilon value
    "EPS_DECAY": 0.995,             # Epsilon decay rate
    "Q_NET_Hidden_Dims": (64, 64)   # Size of the hidden layer in Q-Net
}


def train(_env, _agent, _brain_name):

    # watch an untrained agent
    env_info = _env.reset(train_mode=False)[_brain_name]
    state = env_info.vector_observations[0]
    score = 0  # initialize the score
    for _ in range(50):
        action = _agent.act(state)  # select an action
        env_info = _env.step(action)[_brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break
    print("UnTrained Agent's Score: {}".format(score))


    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = config["EPS_START"]  # initialize epsilon

    for i_episode in range(1, config["N_EPISODS"] + 1):

        env_info = _env.reset(train_mode=True)[_brain_name]
        state = env_info.vector_observations[0]
        score = 0

        while True:
            action = _agent.act(state, eps)

            env_info = _env.step(action)[_brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            _agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(config["EPS_END"], config["EPS_DECAY"] * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(_agent.qnetwork_local.state_dict(), 'results/checkpoint.pth')
            break

    return scores, _agent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=bool, default=True, help='Train otherwise Test/Eval')
    args = parser.parse_args()

    env = UnityEnvironment(file_name="../Banana_Linux/Banana.x86_64", seed=config["SEED"])
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size=state_size, action_size=action_size, device=device, config=config)

    if args.is_training:
        scores, agent = train(env, agent, brain_name)

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig('results/learning_curve.png')
        plt.show()

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('results/checkpoint.pth'))

    # watch an trained agent
    score = 0  # initialize the score
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    while True:
        action = agent.act(state)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break
    print("Trained Agent's Score: {}".format(score))

    env.close()