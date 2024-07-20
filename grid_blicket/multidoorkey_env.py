import random
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np

class MultiDoorKeyEnv(MiniGridEnv):
    def __init__(self,
                size=8,
                agent_start_pos=(1, 1),
                agent_start_dir=0,
                n_keys=1,
                max_steps=None,
                fixed_positions=False,  # If True, the key and door positions are fixed during env.reset()
                **kwargs,
                ):        
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir
            self.n_keys = n_keys
            self.size = size
            self.door_states = [False] * n_keys
            self.fixed_positions = fixed_positions
            self.key_positions = None
            self.door_positions = None
            self.to_reward = False
            self.rewarded = False  # makes sure agent only gets rewarded once

            if max_steps is None:
                max_steps = (n_keys+1) * 4 * size**2
            
            mission_space = MissionSpace(mission_func=self._gen_mission)

            super().__init__(
                mission_space=mission_space,
                grid_size=size,
                see_through_walls=True,
                max_steps=max_steps,
                **kwargs,
            )

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the goal square
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.agent_pos = self.place_agent()

        # List of all possible positions except the outer walls
        all_positions = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]
        all_positions.remove(self.agent_pos)
        all_positions.remove((width - 2, height - 2))

        if not self.fixed_positions or (self.key_positions is None and self.door_positions is None):
            random.shuffle(all_positions)
            
            # Randomly place the keys
            self.key_positions = [all_positions.pop() for _ in range(self.n_keys)]
            
            # Randomly place the corresponding doors
            self.door_positions = [all_positions.pop() for _ in range(self.n_keys)]

        # Place keys and doors
        for i, pos in enumerate(self.key_positions):
            self.grid.set(pos[0], pos[1], Key(COLOR_NAMES[i]))

        for i, pos in enumerate(self.door_positions):
            self.grid.set(pos[0], pos[1], Door(COLOR_NAMES[i], is_locked=True))

        self.mission = "use the key to open the door and then get to the goal"

    def _reward(self):
        # Reward is 10 if reaches goal, step penalty -0.01
        if self.to_reward and not self.rewarded:
            return 10.0  # High reward for reaching the goal after opening all doors
        else:
            return -0.01  # Small negative reward (step penalty) for each action

    def reset(self, *, seed=None, options=None, fixed_positions=None):
        if fixed_positions is not None:
            self.fixed_positions = fixed_positions
        
        if not self.fixed_positions:
            self.key_positions = None
            self.door_positions = None

        self.door_states = [False] * self.n_keys
        self.to_reward = False
        self.rewarded = False
        
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.step_count += 1
        reward = self._reward()
        truncated = False
        terminated = False
        # Execute the action
        # observation, reward, terminated, truncated, info = MiniGridEnv.step(self, action)

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                # terminate only if all doors have been opened
                if all(self.door_states):
                    self.to_reward = True
                    reward = self._reward()
                    self.rewarded = True

        # Pick up an object. Used on keys
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])  # Place the object out of the grid
                    self.grid.set(fwd_pos[0], fwd_pos[1], None) # Remove the key from the grid
        # Drop an object. Used on keys
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)  # Place the key on the grid
                self.carrying.cur_pos = fwd_pos  # Set the key's position
                self.carrying = None

        # Toggle/activate an object. Used on doors
        elif action == self.actions.toggle:
            if fwd_cell is not None and fwd_cell.type == 'door':
                if fwd_cell.is_locked:
                    # Check if the agent has the corresponding key
                    if self.carrying and self.carrying.color == fwd_cell.color:
                        fwd_cell.is_locked = False
                        fwd_cell.is_open = True
                        self.door_states[COLOR_NAMES.index(fwd_cell.color)] = True
                        self.carrying = None  # Drop the key after opening the door
        
        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        # Check if the max number of steps is reached
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
        terminated = self.rewarded

        return obs, reward, terminated, truncated, {}

# Function to visualize the environment
def visualize_episode(env, max_steps=100):
    obs = env.reset()
    for step in range(max_steps):
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)

        # Render the environment
        img = env.render()
        plt.imshow(img)
        plt.axis('off')
        display(plt.gcf())
        clear_output(wait=True)
        if terminated or truncated:
            break

def main():
    # Create the environment
    env = MultiDoorKeyEnv(render_mode='rgb_array', fixed_positions=True)
    # Visualize an episode with a random agent
    visualize_episode(env)

if __name__ == '__main__':
    main()