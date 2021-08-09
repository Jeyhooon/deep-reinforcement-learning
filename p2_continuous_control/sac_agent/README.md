[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Double-Jointed Arm Reacher (Continuous Control)

![Trained Agent][image1]

### Project Details
This project is about training a policy-gradient-based agent that tries to follow the target locations. \
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. \
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, the Unity environment with single agent is used and trained using data efficient off-policy policy gradient algorithm; [soft-actor-critic (SAC)](https://arxiv.org/abs/1801.01290). 
SAC tries to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.
### Getting Started

Clone this repository and install the requirements in a [virtual-env](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) using `pip install requirements.txt`

### Instructions
Run `main.py` to train the agent; hyperparameters can be changed inside the `main.py` script (code is adapted from: [mimoralea/gdrl](https://github.com/mimoralea/gdrl)):
```Python
config = {
    "ROOT_DIR": "results",                  # directory to save the results
    "BUFFER_SIZE": int(1e6),                # replay buffer size
    "BATCH_SIZE": 256,                      # mini-batch size
    "WARMUP_BATCHES": 10,                   # number of initial batches to fill the buffer with
    "TAU": 1e-3,                            # for soft update of target parameters
    "UPDATE_EVERY": 1,                      # how often to update the network
    "SEED": [1],                            # list of the seed to do randomize each training
    "Q_NET_Hidden_Dims": (128, 128),        # Size of the hidden layer in Q-Net
    "Q_LR": 7e-4,                           # Q-Net learning rate
    "Q_MAX_GRAD_NORM": float(100.0),        # to clip gradients of Q-Net
    "POLICY_NET_Hidden_Dims": (64, 64),     # Size of the hidden layer in Policy-Net
    "POLICY_LR": 5e-4,                      # Policy-Net learning rate
    "POLICY_MAX_GRAD_NORM": float(100.0),   # to clip gradients of the Policy-Net

    "ENV_SETTINGS": {
            'ENV_NAME': '../data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64',
            'GAMMA': 0.99,
            'MAX_MINUTES': 300,
            'MAX_EPISODES': 10000,
            'GOAL_MEAN_100_REWARD': 30
        }
}
```
In case not want to train the agent but to watch the smart agent, run `main.py` with `--is_training False` argument.

Please refer to `Report.ipynb` for learning more about the implementation details.