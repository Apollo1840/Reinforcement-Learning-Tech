import time
import matplotlib.pyplot as plt
from tqdm import tqdm


class Play():

    def __init__(self, env, strategy, num_steps=100, reward_mapping=None):
        self.env = env
        self.strategy = strategy
        self.num_steps = num_steps
        self.reward_mapping = reward_mapping  # Optional: mapping from rewards to custom values (e.g., for FrozenLakeV1)

    def run(self, num_episodes=30, is_render=True, verbose=True):

        # Track the rewards per episode (optional)
        total_rewards = []

        for episode in tqdm(range(num_episodes)):
            # Reset the environment to the initial state at the start of each episode
            state = self.env.reset()[0]  # Extract the actual state from the reset

            if is_render:
                # Render the initial state (not in the FrozenLakeV1 class but showing how to work with Q-Learning)
                self.env.render(step_number=0)

            action = self.strategy.act(state)  # Select action based on the current state

            total_reward = 0  # Initialize the reward tracker for this episode

            for step in range(self.num_steps):
                next_state, reward, done, truncated, info = self.env.step(action)  # Apply the action to the environment

                if self.reward_mapping:
                    reward = self.reward_mapping(next_state, reward, done, truncated, info)

                # Update the Q-Learning state-action matrix
                next_action = self.strategy.step(state, action, reward, next_state, done)

                if is_render:
                    # Render the updated environment after each action (optional for multiple episodes)
                    self.env.render(step_number=step + 1, episode_number=episode + 1)

                # Transition to the next state
                state = next_state
                action = next_action

                total_reward += reward  # Accumulate the reward

                if verbose:
                    if done:
                        print(f"Episode {episode + 1} done in {step + 1} steps with total reward: {total_reward}")
                        time.sleep(1)
                    if truncated:
                        print(f"Episode {episode + 1} finished in {step + 1} steps with total reward: {total_reward}")
                        time.sleep(1)

                if done or truncated:
                    break

            # Log the total reward for this episode
            total_rewards.append(total_reward)

        # Close the environment after all episodes
        self.env.close()

        self.plot_learning_curve(total_rewards)

        return total_rewards

    def plot_learning_curve(self, total_rewards):
        # Plot the total reward over time to see the agent's improvement
        plt.plot(total_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode in FrozenLake')
        plt.show()

    def showcase(self, num_steps=100):
        # Reset the environment to the initial state
        state = self.env.reset()[0]

        # Render the initial state (not in the FrozenLakeV1 class but showing how to work with Q-Learning)
        self.env.render(step_number=0)
        action = self.strategy.act(state)  # Select action based on the current state

        # Play the game by following the Q-Learning strategy
        for step in range(num_steps):

            next_state, reward, done, truncated, info = self.env.step(action)  # Apply the action to the environment

            # Render the updated environment after each action
            self.env.render(step_number=step + 1)

            # Update the Q-Learning state-action matrix
            next_action = self.strategy.act(state)

            # Transition to the next state
            state = next_state
            action = next_action

            if done or truncated:
                print("Game Over!")
                time.sleep(1)
                break

        # Close the environment
        self.env.close()

    # for tutorial
    def run_single(self, num_steps=100):

        # Reset the environment to the initial state
        state = self.env.reset()[0]

        # Render the initial state (not in the FrozenLakeV1 class but showing how to work with Q-Learning)
        self.env.render(step_number=0)
        action = self.strategy.act(state)  # Select action based on the current state

        # Play the game by following the Q-Learning strategy
        for step in range(num_steps):

            next_state, reward, done, truncated, info = self.env.step(action)  # Apply the action to the environment

            # Render the updated environment after each action
            self.env.render(step_number=step + 1)

            # Update the Q-Learning state-action matrix
            next_action = self.strategy.step(state, action, reward, next_state, done)

            # Transition to the next state
            state = next_state
            action = next_action

            if done or truncated:
                print("Game Over!")
                time.sleep(1)
                break

        # Close the environment
        self.env.close()
