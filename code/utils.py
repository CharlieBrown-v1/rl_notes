import gym
import pynvml
import numpy as np


from tqdm import tqdm


class Algorithm:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def take_action(self, observation: np.ndarray, deterministic: bool = False):
        raise NotImplementedError

    def train(self, buffer_dict: dict) -> None:
        raise NotImplementedError
    

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
    for idx in tqdm(range(num_episodes)):
        done = False
        obs = env.reset()
        while not done:
            action = agent.take_action(obs)
            next_obs, reward, done, info = env.step(action)
            returns_sum += reward
            obs = next_obs
    
    returns_mean = returns_sum / num_episodes
    
    return returns_mean
