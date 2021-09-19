
import os
import os.path
import wandb

from unityagents import UnityEnvironment
from scripts.utils import *

# for saving gif
import imageio
from tensorboardX import SummaryWriter
dtype=torch.float32

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
RESULTS_DIR = os.path.join('results')
SEEDS = (1)

env = UnityEnvironment(file_name="data/Tennis_Linux_NoVis/Tennis")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
print('The state for the second agent looks like:', states[1])

for i in range(1):                                         # play game for some episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def main():
    num_agents = 2

    seeding()
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 10000
    batchsize = 2000
    # how many episodes to save policy and gif
    save_interval = 1000
    t = 0

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.99999

    # how many episodes before update
    episode_per_update = 5

    log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"

    os.makedirs(model_dir, exist_ok=True)

    buffer = ReplayBuffer(int(1e6))

    # initialize policy and critic
    maddpg = SelfPlayDDPG()
    wandb.watch(maddpg, log="all")

    logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []

    # training loop
    # show progressbar
    import progressbar as pb
    widget = ['episode: ', pb.Counter(), '/', str(number_of_episodes), ' ',
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ']

    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

    # use keep_awake to keep workspace from disconnecting
    for episode in range(number_of_episodes):

        timer.update(episode)

        reward_this_episode = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # get the current state (for each agent) [2, 24]

        # for calculating rewards for this particular episode - addition of all time steps

        # save info or not
        save_info = (episode % save_interval == 0 or episode == number_of_episodes)

        # frames = []
        # tmax = 0
        #         if save_info:
        #             frames.append(env.render('rgb_array'))

        while True:
            t += 1

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(torch.tensor(states[None, ...], dtype=dtype, device=device), noise=noise)
            noise *= noise_reduction

            actions_for_env = torch.cat(actions, dim=0).detach().numpy()  # [num_agents, acts_size]

            # step forward one frame
            env_info = env.step(actions_for_env)[brain_name]
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished

            # add data to buffer
            transition = (states, actions_for_env, rewards, next_states, dones)

            buffer.push(transition)

            reward_this_episode += rewards
            agent0_reward.append(rewards[0])
            agent1_reward.append(rewards[1])

            states = next_states.copy()

            if any(dones):
                break

            # save gif frame
        #             if save_info:
        #                 frames.append(env.render('rgb_array'))
        #                 tmax += 1

        # update once after every episode_per_update
        if len(buffer) > batchsize and episode % episode_per_update == 0:
            samples = buffer.sample(batchsize)
            maddpg.update(samples, logger)
            maddpg.update_targets()  # soft update the target network towards the actual networks

        if episode % 100 == 0 or episode == number_of_episodes - 1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
            agent0_reward = []
            agent1_reward = []
            agent_mean_rew_dict = {}
            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)
                print(f'agent{a_i}/mean_episode({episode})_rewards: {avg_rew}')
                agent_mean_rew_dict.update({f'agent_{a_i + 1}_mean_reward': avg_rew, 'episode': episode})
            wandb.log(agent_mean_rew_dict)

        # saving model
        save_dict_list = []
        if save_info:
            save_dict = {'actor_params': maddpg.ddpg_agent.actor.state_dict(),
                         'actor_optim_params': maddpg.ddpg_agent.actor_optimizer.state_dict(),
                         'critic1_params': maddpg.ddpg_agent.critic_1.state_dict(),
                         'critic1_optim_params': maddpg.ddpg_agent.critic_1_optimizer.state_dict(),
                         'critic2_params': maddpg.ddpg_agent.critic_2.state_dict(),
                         'critic2_optim_params': maddpg.ddpg_agent.critic_2_optimizer.state_dict()
                         }
            save_dict_list.append(save_dict)

            torch.save(save_dict_list, os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

    #             # save gif files
    #             imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)),
    #                             frames, duration=.04)

    env.close()
    logger.close()
    timer.finish()


if __name__ == '__main__':
    device = 'cpu'

    wandb_id = wandb.util.generate_id()
    date_time_now = get_date_time_now()
    wandb.init(project="MADDPG_Tennis", id=wandb_id)
    wandb.run.name = date_time_now + "__" + wandb_id
    wandb.run.save()

    main()