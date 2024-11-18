import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class PolicyGradient:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n

        self.learning_rate = learning_rate
        self.gamma = gamma

        # Build the policy network
        self.model = Sequential([
            Dense(24, input_dim=self.state_space, activation='relu'),
            Dense(self.action_space, activation='softmax')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.memory = []  # Store (state, action, reward)

    def act(self, state):
        """Chooses an action based on the policy network."""
        state = state.reshape([1, self.state_space])  # Ensure correct input shape
        probs = self.model(state).numpy()[0]
        action = np.random.choice(self.action_space, p=probs)
        return action

    def predict(self, state):
        state = state.reshape([1, self.state_space])  # Ensure correct input shape
        probs = self.model(state).numpy()[0]
        return np.argmax(probs)

    def step(self, state, action, reward):
        """Stores the transition."""
        self.memory.append((state, action, reward))

    def train(self):
        """Updates the policy network based on stored experiences."""
        states, actions, rewards = zip(*self.memory)
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Compute discounted rewards
        discounted_rewards = self._discount_rewards(rewards)

        # Compute gradients and update policy
        with tf.GradientTape() as tape:
            action_probs = self.model(states)
            selected_action_probs = tf.reduce_sum(
                tf.one_hot(actions, self.action_space) * action_probs, axis=1
            ) + 1e-10
            loss = -tf.reduce_mean(tf.math.log(selected_action_probs) * discounted_rewards)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.memory = []  # Clear memory after training

    def _discount_rewards(self, rewards):
        """Computes discounted rewards."""
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[t]
            discounted_rewards[t] = cumulative
        return (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)


# Training the Policy Gradient agent on FrozenLake
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False)  # Use deterministic environment for simplicity
    state_space = env.observation_space.n
    action_space = env.action_space.n

    # Create the agent
    agent = PolicyGradient(state_space, action_space)

    episodes = 1000
    success_rate = []

    for episode in range(episodes):
        state = env.reset()
        state = np.eye(state_space)[state]  # One-hot encode state
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.eye(state_space)[next_state]  # One-hot encode next state
            agent.step(state, action, reward)
            state = next_state
            total_reward += reward

            if done:
                agent.train()
                success_rate.append(total_reward)
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Average Reward (Last 100): {np.mean(success_rate[-100:])}")

    env.close()
