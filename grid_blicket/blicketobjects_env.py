import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import miniworld
from miniworld.entity import Entity
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
        self.episode_log = []

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
        # self.key = self.place_entity(Key(color=self.np_random.choice(['red', 'blue', 'purple', 'yellow'])))

        # Place the agent
        self.place_agent()

        self.objects_at_goal = set()
        self.episode_log = []

    def step(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1
        reward = -0.01
        termination = False
        truncation = False
        info = {}
        step_log = ""

        rand = self.np_random if self.domain_rand else None
        fwd_step = self.params.sample(rand, "forward_step")
        fwd_drift = self.params.sample(rand, "forward_drift")
        turn_step = self.params.sample(rand, "turn_step")

        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_back:
            self.move_agent(-fwd_step, fwd_drift)

        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)

        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)

        # Pick up an object
        elif action == self.actions.pickup:
            # Position at which we will test for an intersection
            test_pos = self.agent.pos + self.agent.dir_vec * 1.5 * self.agent.radius
            ent = self.intersect(self.agent, test_pos, 1.2 * self.agent.radius)
            if not self.agent.carrying:
                if isinstance(ent, Entity):
                    if not ent.is_static:
                        self.agent.carrying = ent
                        ##################
                        step_log += f"Step {self.step_count}: "
                        step_log += f"Agent picked up {ent.__class__.__name__}"

        # Drop an object being carried
        elif action == self.actions.drop:
            if self.agent.carrying:
                ent = self.agent.carrying
                self.agent.carrying.pos[1] = 0
                self.agent.carrying = None
                #####################
                if self.near(self.goal_box):
                    self.objects_at_goal.add(ent)
                    step_log += f"Step {self.step_count}: "
                    step_log += f"Agent dropped {ent.__class__.__name__} at goal"
                    print(f"Agent dropped {ent.__class__.__name__} at goal!")
                else:
                    step_log += f"Step {self.step_count}: "
                    step_log += f"Agent dropped {ent.__class__.__name__}"

        # If we are carrying an object, update its position as we move
        if self.agent.carrying:
            ent_pos = self._get_carry_pos(self.agent.pos, self.agent.carrying)
            self.agent.carrying.pos = ent_pos
            self.agent.carrying.dir = self.agent.dir

        # Generate the current camera image
        obs = self.render_obs()

        # Check if all objects are at the goal
        if len(self.objects_at_goal) == 2:
            step_log += f"Step {self.step_count}: "
            step_log += "All objects at goal!"
            print("All objects at goal!")
            reward = 10.0
            termination = True
            truncation = False
        
        # If the maximum time step count is reached
        elif self.step_count >= self.max_episode_steps:
            termination = False
            truncation = True

        
        if step_log != "":
            self.episode_log.append(step_log)
        info['episode_log'] = self.episode_log

        return obs, reward, termination, truncation, info
    

    def reset(self, **kwargs):
        self.episode_log = []  # Reset log at the start of each episode
        return super().reset(**kwargs)



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

def debug_blicket_objects_env(n_episodes=10, max_steps=500):
    # Create the environment
    env = BlicketObjectsEnv(size=10, max_episode_steps=max_steps)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # set up arrays for logging
    rewards_arr = np.zeros((n_episodes, max_steps+1))
    actions_arr = np.zeros((n_episodes, max_steps+1))
    terminated_arr = np.zeros((n_episodes, max_steps+1))
    truncated_arr = np.zeros((n_episodes, max_steps+1))
    total_rewards_arr = np.zeros(n_episodes)


    for i_episode in range(n_episodes):
        print(f"\nEpisode {i_episode + 1}")
        print("===============")
    
        # Reset the environment
        obs = env.reset()
        
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            # Choose a random action
            action = env.action_space.sample()
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update total reward and step count
            total_reward += reward
            step += 1
            
            # Log step information
            rewards_arr[i_episode, step] = reward
            actions_arr[i_episode, step] = action
            terminated_arr[i_episode, step] = int(terminated)
            truncated_arr[i_episode, step] = int(truncated)
            
            # Check if the episode is done
            done = terminated or truncated
        total_rewards_arr[i_episode] = total_reward
        print("\nEpisode finished!")
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward:.2f}")
        if total_reward > -0.001*max_steps:
            print(f"  Log: {env.episode_log}")
        
    # Close the environment
    env.close()

if __name__ == '__main__':
    # main()
    debug_blicket_objects_env(n_episodes=10, max_steps=500)

