import gym
import numpy as np

from multiprocessing import Pool


class VecEnv(gym.Env):
    def __init__(self,
                 id: str,
                 env_num: int = 1,
                 ) -> None:
        super().__init__()
        
        self.env_num = env_num
        self.env_list = [gym.make(id) for _ in range(self.env_num)]
        self.pool = Pool(processes=env_num)
    
    def sample_action(self) -> np.ndarray:
        action_list = []
        for env_idx in range(self.env_num):
            action = self.env_list[env_idx].action_space.sample()
            action_list.append(action)
        action_arr = np.array(action_list)
        
        return action_arr
    
    def reset(self) -> np.ndarray:
        vec_obs_list = []
        for env in self.env_list:
            obs = env.reset()
            vec_obs_list.append(obs)
        vec_obs_arr = np.array(vec_obs_list)
        
        return vec_obs_arr
    
    def parallel_step(self, env_idx: int, action: np.ndarray) -> tuple:
        env = self.env_list[env_idx]
        
        next_obs, reward, done, info = env.step(action)
        
        return next_obs, reward, done, info
    
    def step(self, vec_action: np.ndarray) -> tuple:
        assert vec_action.shape[0] == self.env_num
        step_return_list = self.pool.starmap(self.parallel_step, zip(range(self.env_num), vec_action))
        step_return_arr = np.array(step_return_list, dtype=object)
        
        vec_obs = np.array(np.c_[step_return_arr[:, 0]])
        vec_reward = np.c_[step_return_arr[:, 1]]
        vec_done = np.c_[step_return_arr[:, 2]]
        vec_info = np.c_[step_return_arr[:, 3]]
        
        return vec_obs, vec_reward, vec_done, vec_info
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict


if __name__ == '__main__':
    test_env_id = 'CartPole-v0'
    test_env_num = 2
    vec_env = VecEnv(test_env_id, test_env_num)
    
    for _ in range(10):
        done = False
        vec_obs = vec_env.reset()
        while not done:
            vec_action = vec_env.sample_action()
            vec_obs, reward, done, info = vec_env.step(vec_action)
        print(f'Finish 1 trajectory!')
            