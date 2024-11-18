import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import gym


class A2C:
    def __init__(self, state_space, action_space, gamma=0.99, actor_lr=0.001, critic_lr=0.005):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma

        # Actor and Critic Networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def build_actor(self):
        """Build the actor network."""
        inputs = Input(shape=(self.state_space,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.action_space, activation='softmax')(x)  # Output probabilities
        return Model(inputs, outputs)

    def build_critic(self):
        """Build the critic network."""
        inputs = Input(shape=(self.state_space,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)  # Output value estimate
        return Model(inputs, outputs)

    def act(self, state):
        """Sample an action from the policy."""
        state = state.reshape([1, self.state_space])
        probabilities = self.actor(state).numpy()[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def train(self, state, action, reward, next_state, done):
        """Train both the actor and critic."""
        state = state.reshape([1, self.state_space])
        next_state = next_state.reshape([1, self.state_space])

        # Compute TD target and advantage
        with tf.GradientTape(persistent=True) as tape:
            value = self.critic(state)
            next_value = self.critic(next_state)
            target = reward + self.gamma * next_value * (1 - int(done))
            advantage = target - value

            # Actor loss
            action_probs = self.actor(state)
            action_onehot = tf.one_hot([action], self.action_space)
            selected_action_prob = tf.reduce_sum(action_probs * action_onehot, axis=1)
            actor_loss = -tf.math.log(selected_action_prob + 1e-10) * tf.stop_gradient(advantage)

            # Critic loss (Mean Squared Error)
            critic_loss = tf.reduce_mean(tf.square(target - value))

        # Update Actor and Critic networks
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        del tape


if __name__ == '__main__':
    env = FrozenLakeEnvJP(is_slippery=True)
    strategy = A2C(state_space, action_space)

    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = strategy.act(state)
            next_state, reward, done, _ = env.step(action)

            # Train the agent
            strategy.train(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
                break

    env.close()
