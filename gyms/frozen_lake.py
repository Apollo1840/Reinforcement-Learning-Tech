import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches
from PIL import Image
from IPython.display import display, clear_output
import time

class FrozenLakeV1():

    def __init__(self):
        self.env = gym.make('FrozenLake-v1', is_slippery=True)

        # Load the custom images
        self.player_img = Image.open("material/frozen_lake/player.jpg").resize((40, 40))
        self.hole_img = Image.open("material/frozen_lake/hole.jpg").resize((40, 40))
        self.entrance_img = Image.open("material/frozen_lake/entrance.jpg").resize((40, 40))
        self.goal_img = Image.open("material/frozen_lake/goal.jpg").resize((40, 40))

    def render(self, step_number):
        # Get the grid layout (env.desc stores the grid description)
        grid_array = np.array(self.env.desc, dtype='str')

        # Get the agent's position (env.s gives the current state as a flat index)
        player_position = self.env.s

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
        fig.suptitle(f"Step: {step_number}", y=1.05, fontsize=16)

        # Show the plot
        # plt.show()

        display(plt.gcf())
        clear_output(wait=True)
        plt.close()
