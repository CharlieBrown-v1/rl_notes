import gym
import pynvml
import numpy as np
import torch as th

from torch import nn
from torch.optim import Adam
from typing import Tuple


# PG-based Algorithm
class PGAlgorithm:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        lr: float = 1e-3,
        latent_dim: int = 64,
        device: th.device = 'cuda',
        ) -> None:
        self.env = env
        self.gamma = gamma
        self.latent_dim = latent_dim
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_space_bd = None
        try:
            self.action_size = self.env.action_space.n
            self.action_dim = 1
            self.action_type = 'discrete'
        except AttributeError:
            self.action_size = np.inf
            self.action_dim = self.env.action_space.shape[0]
            self.action_type = 'continuous'
            assert np.all(np.abs(self.env.action_space.low) == np.abs(self.env.action_space.high))
            self.action_space_bd = self.env.action_space.high
        self.device = device
        self.policy_net = PolicyNet(
            observation_dim=self.observation_dim,
            action_size=self.action_size,
            action_dim=self.action_dim,
            action_type=self.action_type,
            latent_dim=self.latent_dim,
            device=self.device,
        )
        
        self.policy_net_optimizer = Adam(self.policy_net.parameters(), lr=lr)
    
    def compute_distribution(self, observation: np.ndarray) -> th.distributions.Distribution:
        observation = th.as_tensor(observation).float().to(self.device)
        
        action_mean, action_std = self.policy_net(observation)
        
        if self.action_type == 'discrete':
            assert action_std is None
            action_dist = th.distributions.Categorical(action_mean)
        elif self.action_type == 'continuous':
            action_dist = th.distributions.Normal(action_mean * self.action_space_bd, action_std)
        else:
            raise NotImplementedError
        
        return action_dist
    
    def take_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action_dist = self.compute_distribution(observation)
        
        if deterministic:
            action = action_dist.mean.cpu().numpy()
        else:
            action = action_dist.sample().cpu().numpy()
        
        clipped_action = action
        if self.action_type == 'continuous':
            clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        return clipped_action

    def train(self, buffer_dict: dict) -> None:
        raise NotImplementedError


# AC-based Algorithm
class ACAlgorihm(PGAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        latent_dim: int = 64,
        device: th.device = 'cuda',
        ) -> None:
        self.env = env
        self.gamma = gamma
        self.latent_dim = latent_dim
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_space_bd = None
        try:
            self.action_size = self.env.action_space.n
            self.action_dim = 1
            self.action_type = 'discrete'
        except AttributeError:
            self.action_size = np.inf
            self.action_dim = self.env.action_space.shape[0]
            self.action_type = 'continuous'
            assert np.all(np.abs(self.env.action_space.low) == np.abs(self.env.action_space.high))
            self.action_space_bd = self.env.action_space.high
        self.device = device
        self.policy_net = PolicyNet(
            observation_dim=self.observation_dim,
            action_size=self.action_size,
            action_dim=self.action_dim,
            action_type=self.action_type,
            latent_dim=self.latent_dim,
            device=self.device,
        )
        self.value_net = ValueNet(
            observation_dim=self.observation_dim,
            latent_dim=self.latent_dim,
            device=self.device,
        )
        
        self.policy_net_optimizer = Adam(self.policy_net.parameters(), lr=actor_lr)
        self.value_net_optimizer = Adam(self.value_net.parameters(), lr=critic_lr)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input: th.Tensor) -> th.Tensor:
        raise NotImplementedError


class PolicyNet(Net):
    def __init__(self,
                 observation_dim: int,
                 action_size: int,
                 action_dim: int,
                 action_type: str,
                 latent_dim: int = 64,
                 device: th.device = 'cuda',
                 ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_size = action_size
        self.action_dim = action_dim
        self.action_type = action_type
        self.latent_dim = latent_dim
        self.device = device
        
        if self.action_type == 'discrete':
            self.mlp = nn.Sequential(
                nn.Linear(self.observation_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.action_size),
                nn.Softmax(dim=-1),
            ).to(self.device)
            self.mean = None
            self.std = None
        elif self.action_type == 'continuous':
            self.mlp = nn.Sequential(
                nn.Linear(self.observation_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.action_dim),
            ).to(self.device)
            self.mean = nn.Tanh()
            self.std = nn.Softplus()
        else:
            raise NotImplementedError

    def forward(self, input: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        latent = self.mlp(input)
        
        if self.action_type == 'discrete':
            assert self.mean is None and self.std is None
            action_mean = latent
            action_std = None
        elif self.action_type == 'continuous':
            action_mean = self.mean(latent)
            action_std = self.std(latent)
        else:
            raise NotImplementedError
        
        return action_mean, action_std


class ValueNet(Net):
    def __init__(self,
                 observation_dim: int,
                 latent_dim: int = 64,
                 device: th.device = 'cuda',
                 ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.device = device
        
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


def eval_policy(env: gym.Env, agent: PGAlgorithm) -> float:
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
