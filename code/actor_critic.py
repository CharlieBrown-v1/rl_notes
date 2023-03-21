import gym
import torch as th
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from utils import Algorithm, PolicyNet, ValueNet
from utils import get_best_cuda


class ActorCritic(Algorithm):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        batch_size: int = 64,
        latent_dim: int = 64,
        device: th.device = 'cuda',
        ) -> None:
        self.env = env        
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.device = device
        
        self.policy_net = PolicyNet(self.env, self.device, self.latent_dim)
        self.value_net = ValueNet(self.env, self.device, self.latent_dim)
        
        self.policy_net_optimizer = Adam(self.policy_net.parameters(), lr=actor_lr)
        self.value_net_optimizer = Adam(self.value_net.parameters(), lr=critic_lr)
   
    def train(self, buffer_dict: dict) -> None:
        obs_tensor = th.as_tensor(buffer_dict['observations']).float().to(self.device)
        next_obs_tensor = th.as_tensor(buffer_dict['next_observations']).float().to(self.device)
        action_tensor = th.as_tensor(buffer_dict['actions']).long().to(self.device).view(-1, self.policy_net.action_dim)
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
            
            action_dist = self.policy_net(sgd_obs_tensor)
            if self.policy_net.action_type == 'discrete':
                log_prob = action_dist.log_prob(sgd_action_tensor.flatten())
            elif self.policy_net.action_type == 'continuous':
                log_prob = action_dist.log_prob(sgd_action_tensor)
            else:
                raise NotImplementedError
            
            value = self.value_net(sgd_obs_tensor)
            next_value = self.value_net(sgd_next_obs_tensor) * (1 - sgd_done_tensor)
            td_target = (sgd_reward_tensor + self.gamma * next_value).detach()
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
    latent_dim = 128
    gamma = 0.99
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
        latent_dim=latent_dim,
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
