import numpy as np
import random

class SARSA:

    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99):
        self.env = env

        # Initialize the state-action matrix (Q-table) with zeros
        self.state_action_matrix = np.zeros((env.observation_space.n, env.action_space.n))
        # self.state_action_matrix = np.random.uniform(0, 1, (env.observation_space.n, env.action_space.n))

        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon (for exploration)
        self.exploration_decay = exploration_decay  # Epsilon decay

    def act(self, state):
        # Epsilon-greedy policy: with probability exploration_rate, choose a random action
        if random.uniform(0, 1) < self.exploration_rate:
            return self.env.action_space.sample()  # Random action
        else:
            # Choose the action with the highest Q-value  for the current state
            return self.predict(state)

    def predict(self, state):
        return np.argmax(self.state_action_matrix[state, :])

    def step(self, state, action, reward, next_state, done):
        next_action = self.act(state)

        td_target = reward + self.discount_factor * self.state_action_matrix[next_state, next_action]
        td_delta = td_target - self.state_action_matrix[state, action]
        self.state_action_matrix[state, action] += self.learning_rate * td_delta

        # Decay exploration rate
        if done and reward>0:
            self.exploration_rate *= self.exploration_decay

        return next_action

