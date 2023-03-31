import gym
import torch as th
import numpy as np

from tqdm import tqdm
from utils import PGAlgorithm
from utils import get_best_cuda


class REINFORCE(PGAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        lr: float = 1e-3,
        latent_dim: int = 64,
        device: th.device = 'cuda',
        update_method: str = 'reward-to-go',
        ) -> None:
        super(REINFORCE, self).__init__(
            env=env,
            gamma=gamma,
            lr=lr,
            latent_dim=latent_dim,
            device=device,
        )
        self.update_method = update_method
        assert self.update_method in ['original', 'reward-to-go']
    
    def compute_returns(self, buffer_dict: dict) -> None:
        returns_list = []
        baseline_list = []
        rewards_list = buffer_dict['rewards']
        for tau_idx in range(len(rewards_list)):
            tau_rewards = rewards_list[tau_idx]
            returns_list.append(np.zeros(tau_rewards.shape))
            returns = 0
            for step in reversed(range(tau_rewards.shape[0])):
                returns = self.gamma * returns + tau_rewards[step]
                returns_list[tau_idx][step] = returns
            if self.update_method == 'original':
                returns_list[tau_idx][:] = returns
            elif self.update_method == 'reward-to-go':
                pass
            else:
                raise NotImplementedError
            baseline_list.append(returns)
        if self.update_method == 'original':
            baseline = np.mean(baseline_list)
            for tau_idx in range(len(returns_list)):
                returns_list[tau_idx] -= baseline
        elif self.update_method == 'reward-to-go':
            pass
        else:
            raise NotImplementedError
        buffer_dict.update({
            'returns': returns_list,
        })

    def train(self, buffer_dict: dict) -> None:
        self.compute_returns(buffer_dict)
        obs_list = buffer_dict['observations']
        action_list = buffer_dict['actions']
        returns_list = buffer_dict['returns']
        loss_tensor = th.zeros(len(obs_list)).to(self.device)
        for tau_idx in range(len(obs_list)):
            obs_tensor = th.as_tensor(obs_list[tau_idx]).to(self.device).float()
            action_tensor = th.as_tensor(action_list[tau_idx]).to(self.device).long()
            returns_tensor = th.as_tensor(returns_list[tau_idx]).to(self.device).float()
            action_dist = self.compute_distribution(obs_tensor)
            if self.policy_net.action_type == 'discrete':
                log_prob = action_dist.log_prob(action_tensor.flatten())
            elif self.policy_net.action_type == 'continuous':
                log_prob = action_dist.log_prob(action_tensor)
            else:
                raise NotImplementedError
            
            loss_tensor[tau_idx] = -(log_prob * returns_tensor).sum()
        
        loss = loss_tensor.mean()
        
        self.policy_net_optimizer.zero_grad()
        loss.backward()
        self.policy_net_optimizer.step()


if __name__ == '__main__':
    lr = 1e-3
    num_episodes = 1000
    num_iterations = 10
    num_taus = 10  # 用 mean returns of num_taus 条轨迹近似期望(Q)
    latent_dim = 128
    gamma = 0.99
    log_interval = 10
    seed = 0
    device = f'cuda:{get_best_cuda()}'
    
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(seed)
    th.manual_seed(seed)
    agent = REINFORCE(
        env=env,
        lr=lr,
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
                    for key in buffer_dict.keys():
                        buffer_dict[key].append([])
                    observation = env.reset()
                    done = False
                    while not done:
                        action = agent.take_action(observation)
                        next_observation, reward, done, _ = env.step(action)
                        buffer_dict['observations'][tau_idx].append(observation)
                        buffer_dict['actions'][tau_idx].append(action)
                        buffer_dict['next_observations'][tau_idx].append(next_observation)
                        buffer_dict['rewards'][tau_idx].append(reward)
                        buffer_dict['dones'][tau_idx].append(done)
                        observation = next_observation
                        episode_returns_sum += reward
                    for key in buffer_dict.keys():
                        buffer_dict[key][tau_idx] = np.array(buffer_dict[key][tau_idx]).astype(float)
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
