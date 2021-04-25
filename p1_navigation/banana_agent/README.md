# Banana Lover (Navigation)

### Project Details
This project is about training a DQN agent that collects yellow bananas in a large, square world while avoiding the blue bananas!
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

Clone this repository and install the requirements in a [virtual-env](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) using `pip install requirements.txt`

### Instructions
Run `main.py` to train the agent; hyperparameters can be changed inside the `main.py` script:
```Python
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
```
In case not want to train the agent but to watch the smart agent, run `main.py` with `--is_training False` argument.

Please refer to `Report.ipynb` for learning more about the implementation details.