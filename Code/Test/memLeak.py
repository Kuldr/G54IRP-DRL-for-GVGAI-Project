import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

n_cpu = 300
env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for _ in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=int(1e10))
