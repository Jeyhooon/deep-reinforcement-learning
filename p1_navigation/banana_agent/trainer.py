
import numpy as np
import random
from collections import namedtuple, deque
import torch
from unityagents import UnityEnvironment
from agent import QNetwork, ReplayBuffer, Agent
import matplotlib.pyplot as plt
import pathlib
import os

os.chdir(pathlib.Path(__file__).parent.absolute())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    "BUFFER_SIZE": int(1e5),  # replay buffer size
    "BATCH_SIZE": 64,         # minibatch size
    "GAMMA": 0.99,            # discount factor
    "TAU": 1e-3,              # for soft update of target parameters
    "LR": 5e-4,               # learning rate
    "UPDATE_EVERY": 4,         # how often to update the network
    "SEED": 0,
    "N_EPISODS": 2000,
    "EPS_START": 1.0,
    "EPS_END": 0.01,
    "EPS_DECAY": 0.995
}


def train(_env, _agent, _brain_name):

    # watch an untrained agent
    env_info = _env.reset(train_mode=False)[_brain_name]
    state = env_info.vector_observations[0]
    score = 0  # initialize the score
    while True:
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
            torch.save(_agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    return scores, _agent


if __name__ == "__main__":

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

    # scores, agent = train(env, agent, brain_name)
    #
    # # plot the scores
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(scores)), scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

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