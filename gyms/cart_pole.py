import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleEnvJP(CartPoleEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = None  # Initialize the viewer as None
        self.render_mode = 'rgb_array'  # Set the render mode to 'human'

    def render(self, **kwargs):
        img = super().render()  # Render and get the current frame as an image
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Step: {kwargs['step_number']}")  # Add the step number as a title to the plot

        display(plt.gcf())
        clear_output(wait=True)
