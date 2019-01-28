import gym
import gym_gvgai
import multiprocessing

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

# multiprocess environment
n_cpu = multiprocessing.cpu_count()
env = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl0-v0') for _ in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/")
model.learn(total_timesteps=int(1e6), tb_log_name="1MTimestepRun")
model.save("models/a2c_boulderdash_1M")
env.close()
