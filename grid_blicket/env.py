import gym
import random
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class GridBlicketEnv(MiniGridEnv):
    def __init__(self,
                size=10,
                agent_start_pos=(1, 1),
                agent_start_dir=0,
                max_steps: int = None,
                **kwargs,
            ):
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir

            if max_steps is None:
                max_steps = 4 * size**2
            
            mission_space = MissionSpace(mission_func=self._gen_mission)

            super().__init__(
                mission_space=mission_space,
                grid_size=size,
                max_steps=max_steps,
                **kwargs,
            )
    @staticmethod
    def _gen_mission():
        return "grand mission"


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the goal square
        self.put_obj(Goal(), width - 2, height - 2)

        # List of all possible positions except the outer walls
        all_positions = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]
        random.shuffle(all_positions)

        # Ensure agent start position is not in the list
        all_positions.remove(self.agent_start_pos)

        # Ensure goal position is not in the list
        all_positions.remove((width - 2, height - 2))

        # Randomly place the keys
        key_positions = [all_positions.pop() for _ in range(3)]
        for i, pos in enumerate(key_positions):
            self.grid.set(pos[0], pos[1], Key(COLOR_NAMES[i]))

        # Randomly place the corresponding doors
        door_positions = [all_positions.pop() for _ in range(3)]
        for i, pos in enumerate(door_positions):
            self.grid.set(pos[0], pos[1], Door(COLOR_NAMES[i], is_locked=True))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    def reset(self, **kwargs):
        obs, info = MiniGridEnv.reset(self, **kwargs)
        self.step_count = 0
        self.carrying = None

        # Generate a new grid
        self._gen_grid(self.width, self.height)

        # Return the first observation
        obs = self.gen_obs()
        return obs, info

    def step(self, action):
        self.step_count += 1

        # Execute the action
        observation, reward, terminated, truncated, info = MiniGridEnv.step(self, action)

        # Penalize for each step taken
        reward -= 0.1

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if action == self.actions.toggle:
            if fwd_cell is not None and fwd_cell.type == 'door':
                if fwd_cell.is_locked:
                    # Check if the agent has the corresponding key
                    if self.carrying and self.carrying.color == fwd_cell.color:
                        fwd_cell.is_locked = False
                        reward += 1
                        self.carrying = None  # Drop the key after opening the door
            elif fwd_cell is not None and fwd_cell.type == 'key':
                if not self.carrying:
                    self.carrying = fwd_cell
                    self.grid.set(*fwd_pos, None)  # Remove the key from the grid

        # Check if agent reached the goal
        if fwd_cell is not None and fwd_cell.type == 'goal':
            reward += 10  # Additional reward for reaching the goal
            terminated = True

        # Check if the max number of steps is reached
        if self.step_count >= self.max_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

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
    env = GridBlicketEnv(render_mode='rgb_array')

    # Visualize an episode with a random agent
    visualize_episode(env)

if __name__ == '__main__':
    main()