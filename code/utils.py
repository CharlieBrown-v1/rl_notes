import gym
import pynvml
import numpy as np
import torch as th

from torch import nn


class Algorithm:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def take_action(self, observation: np.ndarray, deterministic: bool = False):
        raise NotImplementedError

    def train(self, buffer_dict: dict) -> None:
        raise NotImplementedError


class Net(nn.Module):
    def __init__(self) -> None:
        raise NotImplementedError
        
    def forward(self, input: th.Tensor) -> th.Tensor:
        raise NotImplementedError


class PolicyNet(Net):
    def __init__(self,
                 env: gym.Env,
                 device: th.device,
                 latent_dim: int = 64,
                 ) -> None:
        self.observation_dim = env.observation_space.shape[0]
        try:
            self.action_size = env.action_space.n
            self.action_dim = 1
            self.action_type = 'discrete'
        except AttributeError:
            self.action_size = np.inf
            self.action_dim = env.action_space.shape[0]
            self.action_type = 'continuous'
            
        self.device = device
        self.latent_dim = latent_dim
        
        if self.action_type == 'discrete':
            self.mlp = nn.Sequential(
                nn.Linear(self.observation_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.action_size),
                nn.Softmax(dim=-1),
            ).to(self.device)
        elif self.action_type == 'continuous':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, input: th.Tensor) -> th.Tensor:
        latent = self.mlp(input)
        
        return latent


class ValueNet(Net):
    def __init__(self,
                 env: gym.Env,
                 device: th.device,
                 latent_dim: int = 64,
                 ) -> None:
        self.observation_dim = env.observation_space.shape[0]
        self.device = device
        self.latent_dim = latent_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.observation_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
        ).to(self.device)

    def forward(self, input: th.Tensor) -> th.Tensor:
        latent = self.mlp(input)
        
        return latent


def get_best_cuda() -> int:
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    
    best_device_index = deviceMemory.argmax().item()
    print(f'best cuda: {best_device_index}')
    
    return best_device_index


def eval_policy(env: gym.Env, agent: Algorithm) -> float:
    num_episodes = 40
    returns_sum = 0.0
    for idx in range(num_episodes):
        done = False
        obs = env.reset()
        while not done:
            action = agent.take_action(obs)
            next_obs, reward, done, info = env.step(action)
            returns_sum += reward
            obs = next_obs
    
    returns_mean = returns_sum / num_episodes
    
    return returns_mean
