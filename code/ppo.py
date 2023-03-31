import gym
import torch as th
import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from vec_env import VecEnv
from vec_env import parallel_sampling
from utils import ACAlgorihm
from utils import get_best_cuda, eval_policy


class PPO(ACAlgorihm):
    def __init__(
        self,
        env: VecEnv,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        latent_dim: int = 64,
        batch_size: int = 64,
        lamda: float = 0.95,
        epsilon: float = 0.2,
        n_epoch: int = 10,
        device: th.device = 'cuda',
        ) -> None:
        super(PPO, self).__init__(
            env=env,
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            latent_dim=latent_dim,
            device=device,
        )
        
        self.batch_size = batch_size
        self.lamda = lamda
        self.epsilon = epsilon
        self.n_epoch = n_epoch
   
    def train(self, buffer_dict: dict) -> None:
        # view 会按照 tau_00, tau_01, ..., tau_10, tau_11, ... 的顺序排列
        obs_tensor = th.as_tensor(buffer_dict['observations']).float().to(self.device).view(-1, self.observation_dim)
        next_obs_tensor = th.as_tensor(buffer_dict['next_observations']).float().to(self.device).view(-1, self.observation_dim)
        done_tensor = th.as_tensor(buffer_dict['dones']).float().to(self.device).view(-1, 1)
        
        # from zwn
        if 'Pendulum' in self.env.envs[0].unwrapped.spec.id:
            buffer_dict['rewards'] = (buffer_dict['rewards'] + 8) / 8
        reward_tensor = th.as_tensor(buffer_dict['rewards']).float().to(self.device).view(-1, 1)

        # op on the type of action must be cautious !
        old_action_dist = self.compute_distribution(obs_tensor)
        if self.policy_net.action_type == 'discrete':
            action_tensor = th.as_tensor(buffer_dict['actions']).long().to(self.device).view(-1, self.action_dim)
            old_log_prob = old_action_dist.log_prob(action_tensor.flatten()).detach()
        elif self.policy_net.action_type == 'continuous':
            action_tensor = th.as_tensor(buffer_dict['actions']).float().to(self.device).view(-1, self.action_dim)
            old_log_prob = old_action_dist.log_prob(action_tensor).detach()
        else:
            raise NotImplementedError
        
        value = self.value_net(obs_tensor)
        next_value = self.value_net(next_obs_tensor) * (1 - done_tensor)
        td_target = (reward_tensor + self.gamma * next_value).detach()
        td_delta = (td_target - value).detach().cpu().numpy()
        buffer_dict.update({
            'td_delta': td_delta,
        })
        
        self.compute_returns_and_advantage(buffer_dict)
        advantage_tensor = th.as_tensor(buffer_dict['advantages']).float().to(self.device).view(-1, 1)
        returns_tensor = td_target  # TD
        # returns_tensor = value + advantage_tensor  # TD
        for _ in range(self.n_epoch):
            idx_arr = np.arange(obs_tensor.shape[0])
            np.random.shuffle(idx_arr)
            for sgd_idx in range(0, idx_arr.shape[0], self.batch_size):
                sgd_idx_arr = idx_arr[sgd_idx: sgd_idx + self.batch_size]
                sgd_obs_tensor = obs_tensor[sgd_idx_arr]
                sgd_action_tensor = action_tensor[sgd_idx_arr]
                sgd_advantage_tensor = advantage_tensor[sgd_idx_arr]
                sgd_old_log_prob = old_log_prob[sgd_idx_arr]
                sgd_returns_tensor = returns_tensor[sgd_idx_arr]
                
                action_dist = self.compute_distribution(sgd_obs_tensor)
                if self.policy_net.action_type == 'discrete':
                    log_prob = action_dist.log_prob(sgd_action_tensor.flatten())
                elif self.policy_net.action_type == 'continuous':
                    log_prob = action_dist.log_prob(sgd_action_tensor)
                else:
                    raise NotImplementedError
                
                ratio = th.exp(log_prob - sgd_old_log_prob)
                surr0 = ratio * sgd_advantage_tensor
                surr1 = th.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * sgd_advantage_tensor
                
                value = self.value_net(sgd_obs_tensor)

                policy_loss = -th.mean(th.min(surr0, surr1))
                value_loss = F.mse_loss(value, sgd_returns_tensor)
                
                self.policy_net_optimizer.zero_grad()
                self.value_net_optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 1)
                self.policy_net_optimizer.step()
                self.value_net_optimizer.step()

    def compute_returns_and_advantage(self, buffer_dict: dict) -> None:
        # reshape 会按照 tau_00, tau_01, ..., tau_10, tau_11, ... 的顺序排列
        done_arr = np.array(buffer_dict['dones']).reshape(-1, 1)
        td_delta_arr = np.array(buffer_dict['td_delta'])
            
        buffer_dict.update({
            'advantages': np.zeros(td_delta_arr.shape),
        })
        advantage = 0
        for tau_idx in reversed(range(td_delta_arr.shape[0])):
            if done_arr[tau_idx]:
                advantage = 0
            advantage = td_delta_arr[tau_idx] + self.gamma * self.lamda * advantage
            buffer_dict['advantages'][tau_idx] = advantage


def serial_training():
    actor_lr = 1e-4
    critic_lr = 1e-3
    num_episodes = 2000
    num_taus = 1
    num_iterations = 10
    batch_size = 256
    latent_dim = 128
    gamma = 0.99
    lamda = 0.95
    epsilon = 0.2
    n_epoch = 10
    log_interval = 10
    seed = 0
    device = f'cuda:{get_best_cuda()}'
    
    env_name = 'CartPole-v0'
    env_name = 'Pendulum-v0'
    env_name = 'Walker2d-v3'

    env = gym.make(env_name)
    env.seed(seed)
    th.manual_seed(seed)
    
    agent = PPO(
        env=env,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        batch_size=batch_size,
        latent_dim=latent_dim,
        gamma=gamma,
        lamda=lamda,
        epsilon=epsilon,
        n_epoch=n_epoch,
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


def parallel_training():
    env_name = 'CartPole-v1'
    num_envs = 4
    env = VecEnv(env_name, num_envs)
    test_env = gym.make(env_name)
    actor_lr = 1e-3
    critic_lr = 1e-2
    total_steps = int(1e6)
    n_steps = 256
    batch_size = 256
    latent_dim = 128
    gamma = 0.99
    lamda = 0.95
    epsilon = 0.2
    n_epoch = 10
    seed = 0
    device = f'cuda:{get_best_cuda()}'
    agent = PPO(
            env=env,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            batch_size=batch_size,
            latent_dim=latent_dim,
            gamma=gamma,
            lamda=lamda,
            epsilon=epsilon,
            n_epoch=n_epoch,
            device=device,
        )

    env.seed(seed)
    test_env.seed(seed)
    th.manual_seed(seed)

    step = 0
    while step < total_steps:
        with tqdm(range(10)) as train_tqdm:
            for _ in train_tqdm:
                buffer_dict = parallel_sampling(env, agent, n_steps)
                agent.train(buffer_dict)
                step += env.num_envs * n_steps
                returns = eval_policy(test_env, agent)
                train_tqdm.set_postfix({'Returns': returns})
                train_tqdm.update(1)


if __name__ == '__main__':
    # serial_training()
    parallel_training()
