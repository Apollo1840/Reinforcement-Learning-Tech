import gym
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class CartPoleV1():

    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')

    def render(self, step_number):
        img = self.env.render()  # Render and get the current frame as an image
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Step: {step_number}")  # Add the step number as a title to the plot
        
         display(plt.gcf())
        clear_output(wait=True)
