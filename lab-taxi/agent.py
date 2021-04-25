import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.01, gamma=1.0, update_rule: str = "sarsa_max"):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - update_rule: update rule for Q value: [sarsa_max, expected_sarsa]
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.update_rule = update_rule
        self.alpha = alpha
        self.gamma = gamma
        
        self.eps = 1.0
        self.eps_min = 0.0
        self.eps_decay_rate = 0.999

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state in self.Q.keys():
            act_prob = np.ones(self.nA)*self.eps/self.nA
            p_optimal = 1 - self.eps + self.eps/self.nA
            act_prob[np.argmax(self.Q[state])] = p_optimal
            action = np.random.choice(self.nA, p=act_prob)
        else:
            action = np.random.randint(self.nA)
        
        return action
    
    def action_prob(self, state):
        if state in self.Q.keys():
            act_prob = np.ones(self.nA)*self.eps/self.nA
            p_optimal = 1 - self.eps + self.eps/self.nA
            act_prob[np.argmax(self.Q[state])] = p_optimal
        else:
            act_prob = np.ones(self.nA)/self.nA
        
        return act_prob

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        current_q = self.Q[state][action]
        
        if done:
            target = reward
            
        else:
            
            if self.update_rule == "sarsa_max":
                target = reward + self.gamma*np.max(self.Q[next_state])
                                               
            elif self.update_rule == "expected_sarsa":
                act_probs = self.action_prob(next_state)
                target = reward + self.gamma*np.dot(self.Q[next_state], act_probs)
            
            else:
                raise ValueError(f"Update Rule is Not Valid: {self.update_rule}")
        
        self.Q[state][action] = current_q + self.alpha*(target - current_q)
        self.eps = max(self.eps*self.eps_decay_rate, self.eps_min)
                                           
    