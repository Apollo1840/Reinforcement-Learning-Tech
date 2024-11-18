from gym.envs.toy_text import FrozenLakeEnv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display, clear_output
import os
from matplotlib.patches import FancyArrowPatch
import time


class FrozenLakeEnvJP(FrozenLakeEnv):
    # Construct the path to the image files
    base_path = os.path.dirname(__file__)
    path_player_img = os.path.join(base_path, "material/frozen_lake/player.jpg")
    path_hole_img = os.path.join(base_path, "material/frozen_lake/hole.jpg")
    path_entrance_img = os.path.join(base_path, "material/frozen_lake/entrance.jpg")
    path_goal_img = os.path.join(base_path, "material/frozen_lake/goal.jpg")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load the custom images
        self.player_img = Image.open(self.__class__.path_player_img).resize((40, 40))
        self.hole_img = Image.open(self.__class__.path_hole_img).resize((40, 40))
        self.entrance_img = Image.open(self.__class__.path_entrance_img).resize((40, 40))
        self.goal_img = Image.open(self.__class__.path_goal_img).resize((40, 40))

    def render(self, **kwargs):
        # Get the grid layout (env.desc stores the grid description)
        grid_array = np.array(self.desc, dtype='str')

        # Get the agent's position (env.s gives the current state as a flat index)
        player_position = self.s

        # Convert the flat position to 2D coordinates (row, col)
        grid_size = grid_array.shape[0]
        player_row = player_position // grid_size
        player_col = player_position % grid_size

        # Create a blank figure
        fig = plt.figure(figsize=(5, 5))

        # Loop through the grid and place images for each tile
        for row in range(grid_size):
            for col in range(grid_size):
                # Create a new set of axes for each tile
                ax = fig.add_axes([col / grid_size, (grid_size - 1 - row) / grid_size, 1 / grid_size, 1 / grid_size])

                # Render different tiles based on the grid content
                if grid_array[row, col] == 'S':  # Start/Entrance
                    ax.imshow(np.ones((40, 40, 3)))
                elif grid_array[row, col] == 'F':  # Frozen tile (safe to walk)
                    ax.imshow(np.ones((40, 40, 3)))  # Render white tile (a blank white plate)
                elif grid_array[row, col] == 'H':  # Hole
                    ax.imshow(self.hole_img)
                elif grid_array[row, col] == 'G':  # Goal
                    ax.imshow(self.goal_img)

                # Hide axis ticks for each grid cell
                ax.set_xticks([])  # Hide x-axis ticks
                ax.set_yticks([])  # Hide y-axis ticks

        # Overlay the player image on the player's position
        player_ax = fig.add_axes(
            [player_col / grid_size, (grid_size - 1 - player_row) / grid_size, 1 / grid_size, 1 / grid_size])
        player_ax.imshow(self.player_img)
        player_ax.set_xticks([])  # Hide x-axis ticks
        player_ax.set_yticks([])  # Hide y-axis ticks

        # Set the step number as the title
        fig.suptitle(f"Episode: {kwargs.get('episode_number', None)}, Step: {kwargs.get('step_number', None)}", y=1.05,
                     fontsize=16)

        display(plt.gcf())
        clear_output(wait=True)
        plt.close()

    def render2(self, state_action_matrix, **kwargs):
        """
        Function to render:
            - the environment map
            - the state value matrix with arrows
        """
        grid_array = np.array(self.desc, dtype='str')
        player_position = self.s
        grid_size = grid_array.shape[0]

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Render the environment map
        render_map(ax1, grid_array, player_position, self.player_img, self.hole_img, self.goal_img)

        # Render the state-action value matrix
        render_state_action_matrix(ax2, state_action_matrix, grid_size)

        # Set the title for the overall plot
        fig.suptitle(f"Episode: {kwargs.get('episode_number', None)}, Step: {kwargs.get('step_number', None)}",
                     fontsize=16)

        # Show the plot
        display(plt.gcf())
        clear_output(wait=True)
        plt.close()

    @staticmethod
    def render_map(ax, grid_array, player_position, player_img, hole_img, goal_img):
        """
        Render the environment map on the given axes.
        """
        grid_size = grid_array.shape[0]
        player_row = player_position // grid_size
        player_col = player_position % grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                cell_ax = ax.inset_axes(
                    [col / grid_size, (grid_size - 1 - row) / grid_size, 1 / grid_size, 1 / grid_size])

                # Render tiles based on the grid content
                if grid_array[row, col] == 'S':  # Start/Entrance
                    cell_ax.imshow(np.ones((40, 40, 3)))
                elif grid_array[row, col] == 'F':  # Frozen tile
                    cell_ax.imshow(np.ones((40, 40, 3)))
                elif grid_array[row, col] == 'H':  # Hole
                    cell_ax.imshow(hole_img)
                elif grid_array[row, col] == 'G':  # Goal
                    cell_ax.imshow(goal_img)

                # Hide axis ticks for each grid cell
                cell_ax.set_xticks([])
                cell_ax.set_yticks([])

        # Overlay the player image
        player_ax = ax.inset_axes(
            [player_col / grid_size, (grid_size - 1 - player_row) / grid_size, 1 / grid_size, 1 / grid_size])
        player_ax.imshow(player_img)
        player_ax.set_xticks([])
        player_ax.set_yticks([])

    @staticmethod
    def render_state_action_matrix(ax, state_action_matrix, grid_size=4):
        """
        Render the state-action value matrix with arrows indicating the best action.
        """
        state_values = np.max(state_action_matrix, axis=1).reshape(grid_size, grid_size)  # max(Q(s, a)) for each state
        cax = ax.imshow(state_values, cmap='viridis', interpolation='none')
        plt.colorbar(cax, ax=ax)  # Add a color bar to indicate value scale

        # Add labels and formatting
        ax.set_title("State Value Matrix (max(Q(s,a)))")
        ax.set_xticks([])
        ax.set_yticks([])

        # Add arrows pointing to the best action
        for state in range(state_action_matrix.shape[0]):
            row = state // grid_size
            col = state % grid_size
            best_action = np.argmax(state_action_matrix[state, :])

            # Coordinates for the center of the cell
            start_x = col
            start_y = row

            # Add an arrow in the direction of the best action
            if best_action == 0:  # Left
                ax.add_patch(FancyArrowPatch((start_x + 0.2, start_y), (start_x - 0.2, start_y), mutation_scale=15,
                                             color='white', lw=2))
            elif best_action == 1:  # Down
                ax.add_patch(FancyArrowPatch((start_x, start_y - 0.2), (start_x, start_y + 0.2), mutation_scale=15,
                                             color='white', lw=2))
            elif best_action == 2:  # Right
                ax.add_patch(FancyArrowPatch((start_x - 0.2, start_y), (start_x + 0.2, start_y), mutation_scale=15,
                                             color='white', lw=2))
            elif best_action == 3:  # Up
                ax.add_patch(FancyArrowPatch((start_x, start_y + 0.2), (start_x, start_y - 0.2), mutation_scale=15,
                                             color='white', lw=2))
