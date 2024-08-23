import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


from blicketobjects_env import BlicketObjectsEnv  # Import your custom environment

def get_env_max_steps(env_id, max_episode_steps, size):
    if env_id == 'BlicketObjectsEnv':
        env = BlicketObjectsEnv(size=size, max_episode_steps=max_episode_steps, render_mode='rgb_array')
    elif env_id.startswith('MiniWorld'):
        env = gym.make(env_id, render_mode='rgb_array')
    else:
        env = gym.make(env_id, render_mode='rgb_array')
    
    max_steps = getattr(env, 'max_episode_steps', max_episode_steps)
    env.close()
    return max_steps

def make_env(env_id, seed, rank, log_dir, allow_early_resets, max_episode_steps, size):
    def _thunk():
        if env_id == 'BlicketObjectsEnv':
            env = BlicketObjectsEnv(size=size, max_episode_steps=max_episode_steps, render_mode='rgb_array')
        elif env_id.startswith('MiniWorld'):
            env = gym.make(env_id, render_mode='rgb_array')
        else:
            env = gym.make(env_id, render_mode='rgb_array')
        
        if log_dir is not None:
            env = Monitor(env,
                          os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)
        return env
    return _thunk

def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  log_dir,
                  device,
                  allow_early_resets,
                  max_episode_steps,
                  size,
                  num_frame_stack=None):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, max_episode_steps, size)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 3:
        envs = VecNormalize(envs, norm_obs=False, norm_reward=False)
        envs = VecTransposeImage(envs)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)

    # Get max_episode_steps using the separate function
    max_steps = get_env_max_steps(env_name, max_episode_steps, size)

    return envs, max_steps


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack
        self.device = device

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, dones, truncated, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, done) in enumerate(dones):
            if done:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, dones, truncated, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()