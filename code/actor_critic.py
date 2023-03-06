import gym
import torch as th
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from utils import Algorithm
from utils import get_best_cuda, eval_policy


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
        
        self.policy = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_size),
            nn.Softmax(dim=-1),
        ).to(self.device)
        self.value = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)
        
        self.policy_optimizer = Adam(self.policy.parameters(), lr=actor_lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=critic_lr)
    
    def take_action(self, observation: th.Tensor, deterministic: bool = False):
        observation = th.as_tensor(observation).float().to(self.device)
        
        probs = self.policy(observation)
        action_dist = th.distributions.Categorical(probs)
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        
        return action.item()

    def compute_returns(self, buffer_dict: dict, valid_count: int = None) -> None:
        assert valid_count is not None
        rewards_arr = buffer_dict['rewards'][:valid_count]
        dones_arr = buffer_dict['dones'][:valid_count]
        returns_arr = np.zeros(rewards_arr.shape)
        returns = 0
        for step in reversed(range(valid_count)):
            if dones_arr[step]:
                returns = 0
            returns = self.gamma * returns + rewards_arr[step]
            returns_arr[step] = returns
        buffer_dict.update({
            'returns': returns_arr,
        })

    def train(self, buffer_dict: dict, valid_count: int = None) -> None:
        if valid_count is None:
            valid_count = buffer_dict['observations'].shape[0]
        self.compute_returns(buffer_dict, valid_count)
        obs_arr = buffer_dict['observations'][:valid_count]
        next_obs_arr = buffer_dict['next_observations'][:valid_count]
        action_arr = buffer_dict['actions'][:valid_count]
        done_arr = buffer_dict['dones'][:valid_count]
        reward_arr = buffer_dict['rewards'][:valid_count]
        returns_arr = buffer_dict['returns'][:valid_count]
        idx_arr = np.arange(obs_arr.shape[0])
        np.random.shuffle(idx_arr)
        
        for sgd_idx in range(0, obs_arr.shape[0], self.batch_size):
            obs_tensor = th.as_tensor(obs_arr).to(self.device).float()[sgd_idx: sgd_idx + self.batch_size]
            next_obs_tensor = th.as_tensor(next_obs_arr).to(self.device).float()[sgd_idx: sgd_idx + self.batch_size]
            action_tensor = th.as_tensor(action_arr).to(self.device).long()[sgd_idx: sgd_idx + self.batch_size]
            done_tensor = th.as_tensor(done_arr).to(self.device).float()[sgd_idx: sgd_idx + self.batch_size]
            reward_tensor = th.as_tensor(reward_arr).to(self.device).float()[sgd_idx: sgd_idx + self.batch_size]
            returns_tensor = th.as_tensor(returns_arr).to(self.device).float()[sgd_idx: sgd_idx + self.batch_size]
            
            logit = self.policy(obs_tensor)
            log_prob = th.log(th.gather(logit, 1, action_tensor.view(-1, 1)))
            value = self.value(obs_tensor) * (1 - done_tensor)
            next_value = self.value(next_obs_tensor) * (1 - done_tensor)
            td_target = reward_tensor + self.gamma * next_value
            td_delta = (td_target - value).detach()

            policy_loss = -(log_prob * td_delta).mean()
            value_loss = F.mse_loss(value, returns_tensor)

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
    

if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_steps = 100000
    num_iterations = 10
    batch_size = 64
    buffer_size = 512
    hidden_dim = 128
    gamma = 0.98
    log_interval = 1000
    seed = 0
    device = f'cuda:{get_best_cuda()}'
    
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    env.seed(seed)
    test_env.seed(seed)
    th.manual_seed(seed)
    
    assert buffer_size >= batch_size
    
    agent = ActorCritic(
        env=env,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        gamma=gamma,
        device=device,
    )
    
    buffer_dict = {
        'observations': np.zeros((buffer_size, agent.observation_dim)),
        'actions': np.zeros((buffer_size, agent.action_dim)),
        'next_observations': np.zeros((buffer_size, agent.observation_dim)),
        'rewards': np.zeros((buffer_size, 1)),
        'dones': np.zeros((buffer_size, 1)),
    }
    for iter in range(num_iterations):
        with tqdm(total=int(num_steps / num_iterations), desc='Iteration %d' % iter) as pbar:
            iter_step = 0
            while iter_step < int(num_steps / num_iterations):
                done = False
                observation = env.reset()
                while not done:
                    action = agent.take_action(observation)
                    next_observation, reward, done, _ = env.step(action)
                    buffer_step = iter_step % buffer_size
                    buffer_dict['observations'][buffer_step] = observation
                    buffer_dict['actions'][buffer_step] = action
                    buffer_dict['next_observations'][buffer_step] = next_observation
                    buffer_dict['rewards'][buffer_step] = reward
                    buffer_dict['dones'][buffer_step] = done
                    observation = next_observation
                    iter_step += 1
                    # train
                    if (iter_step + 1) % buffer_size == 0:
                        agent.train(buffer_dict)
                    # log
                    if (iter_step + 1) % log_interval == 0:
                        pbar.set_postfix({
                            'step':
                            '%d' % (num_steps / log_interval * iter + iter_step + 1),
                            'returns':
                            '%.3f' % eval_policy(test_env, agent),
                        })
                        pbar.update(1)
            valid_count = iter_step % buffer_size
            agent.train(buffer_dict, valid_count)
    print(f'Finish training!')
    