import gym
import concurrent.futures
import numpy as np

from utils import PGAlgorithm
from tqdm import tqdm
from typing import Tuple
from typing import TypeVar, Generic


Array_T = TypeVar('Array_T', bound=np.ndarray)
class Array(Generic[Array_T]):
    def __init__(self, data: Array_T) -> None:
        self.data = data


class VecEnv(gym.Env):
    def __init__(self, env_name: str, num_envs: int):
        super(VecEnv, self).__init__()
        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = [gym.make(self.env_name) for _ in range(self.num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self) -> Array[np.ndarray]:
        return np.array([env.reset() for env in self.envs])

    def sample_action(self) -> Array[np.ndarray]:
        return np.array([env.action_space.sample() for env in self.envs])

    def step(self, actions: Array[np.ndarray]) -> Tuple[Array[np.ndarray], Array[np.ndarray], Array[bool], Array[dict]]:
        next_obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for env, action in zip(self.envs, actions):
                future = executor.submit(env.step, action)
                step_result = future.result()
                next_obs, reward, done, info = step_result
                # When done, V(next_obs) 恒为 0, 因此可以舍弃 next_obs 记录的信息, 用于记录 reset_obs
                if step_result[2]:
                    reset_obs = env.reset()
                    next_obs = reset_obs
                next_obs_list.append(next_obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
        
        next_obs_arr = np.array(next_obs_list)
        reward_arr = np.array(reward_list)
        done_arr = np.array(done_list)
        info_arr = np.array(info_list, dtype=object)
        
        return next_obs_arr, reward_arr, done_arr, info_arr


def test_speed(env: VecEnv, total_steps: int = 10000) -> float:
    import time
    step = 0
    samples_list = []
    env.reset()
    s = time.time()
    while step < total_steps:
        action = env.sample_action()
        samples = env.step(action)
        samples_list.append(samples)
        step += len(env.envs)
    time_cost = time.time() - s
    
    return total_steps / time_cost


def parallel_sampling(env: VecEnv, agent: PGAlgorithm, n_steps: int) -> dict:
    observation_dim = agent.observation_dim
    action_dim = agent.action_dim
    buffer_dict = {
                    'observations': np.zeros((env.num_envs, n_steps, observation_dim)),
                    'actions': np.zeros((env.num_envs, n_steps, action_dim)),
                    'next_observations': np.zeros((env.num_envs, n_steps, observation_dim)),
                    'rewards': np.zeros((env.num_envs, n_steps, 1)),
                    'dones': np.zeros((env.num_envs, n_steps, 1))
                    }
    observation = env.reset()
    for step in range(n_steps):
        action = agent.take_action(observation)
        next_observation, reward, done, _ = env.step(action)
        buffer_dict['observations'][:, step] = observation
        buffer_dict['actions'][:, step] = action.reshape((-1, action_dim))
        buffer_dict['next_observations'][:, step] = next_observation
        buffer_dict['rewards'][:, step] = reward.reshape((-1, 1))
        buffer_dict['dones'][:, step] = done.reshape((-1, 1))
        observation = next_observation  # Step 确保了 done 对应的 next_obs = reset_obs
    
    return buffer_dict


if __name__ == '__main__':
    pass
