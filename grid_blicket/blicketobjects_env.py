import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import miniworld
from miniworld.entity import Box, Key, Ball
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS
import matplotlib.pyplot as plt
import imageio
import os
from PIL import Image

class BlicketObjectsEnv(MiniWorldEnv):
    """
    ## Description

    A custom MiniWorld environment where the agent must collect three objects
    (a ball, a key, and a box) and deliver them to a green goal box in a corner
    of the room. The agent must manually pick up and drop objects.

    ## Action Space

    | Num | Action         |
    |-----|----------------|
    | 0   | turn left      |
    | 1   | turn right     |
    | 2   | move forward   |
    | 3   | move back      |
    | 4   | pickup         |
    | 5   | drop           |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agent sees.

    ## Rewards

    - Each step: -0.01
    - Picking up an object: +0.1
    - Dropping an object at the goal: +1.0
    - Completing the task (all objects at goal): +10.0

    ## Arguments

    ```python
    BlicketObjectsEnv(size=10, max_episode_steps=1000)
    ```

    `size`: size of the square room
    `max_episode_steps`: maximum number of steps per episode

    ## Episode Termination

    The episode terminates if any of the following occurs:

    1. The agent successfully delivers all three objects to the green goal box.
    2. The episode reaches the maximum number of steps.

    """

    def __init__(
        self,
        size=10,
        max_episode_steps=1000,
        **kwargs
    ):
        self.size = size
        super().__init__(
            max_episode_steps=max_episode_steps,
            **kwargs
        )

        # Allow movement actions, pickup, and drop
        self.action_space = spaces.Discrete(self.actions.drop + 1)

    def _gen_world(self):
        # Create a new room
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True
        )

        # Place the green goal box in a corner
        self.goal_box = Box(color='green', size=0.8)
        self.place_entity(
            self.goal_box,
            pos=np.array([self.size - 0.6, 0, self.size - 0.6])
        )

        # Place other objects randomly
        self.box = self.place_entity(Box(color=self.np_random.choice(['red', 'blue', 'purple', 'yellow']), size=0.9))
        self.ball = self.place_entity(Ball(color=self.np_random.choice(['red', 'blue', 'purple', 'yellow']), size=0.9))
        self.key = self.place_entity(Key(color=self.np_random.choice(['red', 'blue', 'purple', 'yellow'])))

        # Place the agent
        self.place_agent()

        self.objects_at_goal = set()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Small punishment for each step
        reward = -0.01

        # Handle pickup action
        if action == self.actions.pickup:
            if self.agent.carrying is None:
                for obj in [self.box, self.ball, self.key]:
                    if self.near(obj):
                        self.agent.carrying = obj
                        reward += 0.1
                        break

        # Handle drop action
        elif action == self.actions.drop:
            if self.agent.carrying is not None:
                # Check if near the goal
                if self.near(self.goal_box):
                    self.objects_at_goal.add(self.agent.carrying)
                    reward += 1.0
                self.agent.carrying = None

        # Check if all objects are at the goal
        if len(self.objects_at_goal) == 3:
            reward += 10.0
            terminated = True

        return obs, reward, terminated, truncated, info


def main():
    # Create a directory for saving frames
    frames_dir = 'episode_frames'
    os.makedirs(frames_dir, exist_ok=True)

    # Create the environment
    env = BlicketObjectsEnv(size=10, max_episode_steps=1000, render_mode='rgb_array')

    obs, info = env.reset()
    frames = []

    done = False
    step = 0
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the frame
        img = env.render()
        
        # Convert to PIL Image and append to frames
        pil_img = Image.fromarray(img)
        frames.append(pil_img)

        # Save individual frame (optional)
        pil_img.save(f"{frames_dir}/frame_{step:03d}.png")
        
        step += 1

    env.close()

    # Save the frames as a GIF
    frames[0].save("blicket_objects_episode.gif", save_all=True, append_images=frames[1:], optimize=False, duration=33, loop=0)

    print("Episode saved as 'blicket_objects_episode.gif'")



if __name__ == '__main__':
    main()

