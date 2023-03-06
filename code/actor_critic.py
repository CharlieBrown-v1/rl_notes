import gym
import torch as th
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from utils import Algorithm
from utils import get_best_cuda


class ActorCritic(Algorithm):
    def __init__(
        self,
        env: gym.Env,
        actor_lr: float,
        critic_lr: float,
        batch_size: int,
        hidden_dim: int,
        gamma: float,
        device: th.device,
        ) -> None:
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = 1
        self.action_size = env.action_space.n
        
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.device = device
        
        self.policy_net = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_size),
            nn.Softmax(dim=-1),
        ).to(self.device)
        self.value_net = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)
        
        self.policy_net_optimizer = Adam(self.policy_net.parameters(), lr=actor_lr)
        self.value_net_optimizer = Adam(self.value_net.parameters(), lr=critic_lr)
    
    def take_action(self, observation: np.ndarray, deterministic: bool = False):
        observation = th.as_tensor(observation).float().to(self.device)
        
        probs = self.policy_net(observation)
        action_dist = th.distributions.Categorical(probs)
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        
        return action.item()

    def train(self, buffer_dict: dict) -> None:
        obs_tensor = th.as_tensor(buffer_dict['observations']).float().to(self.device)
        next_obs_tensor = th.as_tensor(buffer_dict['next_observations']).float().to(self.device)
        action_tensor = th.as_tensor(buffer_dict['actions']).long().to(self.device).view(-1, 1)
        done_tensor = th.as_tensor(buffer_dict['dones']).long().to(self.device).view(-1, 1)
        reward_tensor = th.as_tensor(buffer_dict['rewards']).float().to(self.device).view(-1, 1)
        
        idx_arr = np.arange(obs_tensor.shape[0])
        np.random.shuffle(idx_arr)
        for sgd_idx in range(0, idx_arr.shape[0], self.batch_size):
            sgd_idx_arr = idx_arr[sgd_idx: sgd_idx + self.batch_size]
            sgd_obs_tensor = obs_tensor[sgd_idx_arr]
            sgd_next_obs_tensor = next_obs_tensor[sgd_idx_arr]
            sgd_action_tensor = action_tensor[sgd_idx_arr]
            sgd_done_tensor = done_tensor[sgd_idx_arr]
            sgd_reward_tensor = reward_tensor[sgd_idx_arr]
            
            logit = self.policy_net(sgd_obs_tensor)
            log_prob = th.log(th.gather(logit, 1, sgd_action_tensor))
            value = self.value_net(sgd_obs_tensor)
            next_value = self.value_net(sgd_next_obs_tensor) * (1 - sgd_done_tensor)
            td_target = sgd_reward_tensor + self.gamma * next_value
            td_delta = (td_target - value).detach()

            policy_loss = -(log_prob * td_delta).mean()
            value_loss = F.mse_loss(value, td_target)

            self.policy_net_optimizer.zero_grad()
            self.value_net_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.policy_net_optimizer.step()
            self.value_net_optimizer.step()


if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    num_taus = 10
    num_iterations = 10
    batch_size = 64
    hidden_dim = 128
    gamma = 0.98
    log_interval = 10
    seed = 0
    device = f'cuda:{get_best_cuda()}'
    
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    env.seed(seed)
    test_env.seed(seed)
    th.manual_seed(seed)
    
    agent = ActorCritic(
        env=env,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        gamma=gamma,
        device=device,
    )
    
    returns_list = []
    for iter in range(num_iterations):
        with tqdm(total=int(num_episodes / num_iterations), desc='Iteration %d' % iter) as pbar:
            for i_episode in range(int(num_episodes / num_iterations)):
                episode_returns_sum = 0
                buffer_dict = {
                    'observations': [],
                    'actions': [],
                    'next_observations': [],
                    'rewards': [],
                    'dones': []
                }
                for tau_idx in range(num_taus):
                    observation = env.reset()
                    done = False
                    while not done:
                        action = agent.take_action(observation)
                        next_observation, reward, done, _ = env.step(action)
                        buffer_dict['observations'].append(observation)
                        buffer_dict['actions'].append(action)
                        buffer_dict['next_observations'].append(next_observation)
                        buffer_dict['rewards'].append(reward)
                        buffer_dict['dones'].append(done)
                        observation = next_observation
                        episode_returns_sum += reward
                for key in buffer_dict.keys():
                    buffer_dict[key] = np.array(buffer_dict[key]).astype(float)
                agent.train(buffer_dict)
                returns_list.append(episode_returns_sum / num_taus)
                if (i_episode + 1) % log_interval == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / log_interval * iter + i_episode + 1),
                        'returns':
                        '%.3f' % np.mean(returns_list[-log_interval:])
                    })
                pbar.update(1)

    print(f'Finish training!')
    