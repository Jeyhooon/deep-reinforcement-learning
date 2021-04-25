from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--agent", type=str, default="sarsa_max", help="Update Rule for Learning the Q Values")
args = parser.parse_args()

num_episodes=20000
window=100

env = gym.make('Taxi-v3')
agent = Agent(update_rule=args.agent)
avg_rewards, best_avg_reward = interact(env, agent, num_episodes, window)

env.render()
interact(env, agent, 5, 5)

if best_avg_reward >= 9.1:
    plt.figure()
    plt.plot(np.linspace(0,num_episodes,len(avg_rewards),endpoint=False), np.asarray(avg_rewards))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % window)
    plt.show()