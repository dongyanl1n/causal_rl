import os
import sys
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.box import Box
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper, VecMonitor, VecNormalize)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from functools import partial
sys.path.insert(0, '/home/mila/l/lindongy/causal_rl/grid_blicket')
from multidoorkey_env import MultiDoorKeyEnv

############### code taken from prioritized level replay but adapted to SB3 ###############
class SeededSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)

    def seed_async(self, seed, index):
        self.remotes[index].send(('seed', seed))
        self.waiting = True

    def seed_wait(self, index):
        obs = self.remotes[index].recv()
        self.waiting = False
        return obs

    def seed(self, seed, index):
        self.seed_async(seed, index)
        return self.seed_wait(index)


def get_env_max_steps(env_name):
    if env_name.startswith("MiniGrid"):
        return gym.make(env_name).max_steps
    elif env_name.startswith("MultiDoorKeyEnv"):  # eg. MultiDoorKeyEnv-8x8-3keys-v0
        _, size, n_keys, _ = env_name.split('-') # ['MultiDoorKeyEnv', '8x8', '3keys', 'v0']
        size = int(size.split('x')[0])
        n_keys = int(n_keys.split('keys')[0])
        return MultiDoorKeyEnv(n_keys=n_keys, size=size).max_steps
    else:
        raise ValueError(f"Unknown MiniGrid-based environment {env_name}")
    
class VecMinigrid(SeededSubprocVecEnv):
    def __init__(self, num_envs, env_name, seeds, fully_observed=False):
        if seeds is None:
            seeds = [int.from_bytes(os.urandom(4), byteorder="little") for _ in range(num_envs)]
        env_fn = [partial(self._make_minigrid_env, env_name, seed, fully_observed) for seed in seeds]
        super().__init__(env_fn)

    @staticmethod
    def _make_minigrid_env(env_name, seed, fully_observed):
        if env_name.startswith("MiniGrid"):
            env = gym.make(env_name)
        elif env_name.startswith("MultiDoorKeyEnv"):  # eg. MultiDoorKeyEnv-8x8-3keys-v0
            _, size, n_keys, _ = env_name.split('-') # ['MultiDoorKeyEnv', '8x8', '3keys', 'v0']
            size = int(size.split('x')[0])
            n_keys = int(n_keys.split('keys')[0])
            env = MultiDoorKeyEnv(n_keys=n_keys, size=size)
        else:
            raise ValueError(f"Unknown MiniGrid-based environment {env_name}")

        # env.seed(seed)
        if fully_observed:
            env = RGBImgObsWrapper(env)
        else:
            env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        return env

class VecPyTorchMinigrid(VecEnvWrapper):
    def __init__(self, venv, device, level_sampler=None):
        super().__init__(venv)
        self.device = device
        self.level_sampler = level_sampler
        m, n, c = venv.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(c, m, n), dtype=np.uint8)

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

def make_minigrid_envs(num_envs, env_name, seeds, device, fully_observed, **kwargs):
    ret_normalization = not kwargs.get('no_ret_normalization', False)
    venv = VecMinigrid(num_envs, env_name, seeds, fully_observed=fully_observed)
    venv = VecMonitor(venv=venv, filename=None)
    venv = VecNormalize(venv=venv, norm_obs=False, norm_reward=ret_normalization)
    envs = VecPyTorchMinigrid(venv, device)
    return envs, get_env_max_steps(env_name)

############################################################

# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

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
