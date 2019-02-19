import gym
import gym_gvgai
import multiprocessing
import numpy as np

from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

from CustomPolicy import CustomPolicy
from EnvWrapper import CustomVecEnvWrapper

RENDER_TO_SCREEN = False
stepsUpdate = 5 # 1 to render each frame | Otherwise not really sure why you want it larger
callbacks = 0

def callback(locals, _):
    global callbacks
    callbacks += 1
    if RENDER_TO_SCREEN:
        locals["self"].env.render()
    # Saves the model every 1000 calls
    if callbacks % 1000 == 0:
        locals['self'].save("models/a2c-big-run-" + str(callbacks))
    return True # Returns true as false ends the training

n = 1
list = [lambda: gym.make('gvgai-boulderdash-lvl0-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-boulderdash-lvl1-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-boulderdash-lvl2-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-boulderdash-lvl3-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-boulderdash-lvl4-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-missilecommand-lvl0-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-missilecommand-lvl1-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-missilecommand-lvl2-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-missilecommand-lvl3-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-missilecommand-lvl4-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-aliens-lvl0-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-aliens-lvl1-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-aliens-lvl2-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-aliens-lvl3-v0') for _ in range(n)] + \
       [lambda: gym.make('gvgai-aliens-lvl4-v0') for _ in range(n)]



# multiprocess environment
n_cpu = multiprocessing.cpu_count()
venv = SubprocVecEnv(list)
env = CustomVecEnvWrapper(venv, (130, 260, 3))
model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBigRun/", n_steps=stepsUpdate)
model.learn(total_timesteps=int(1e3), tb_log_name="BigRun", callback=callback)
env.close()
model.save("models/a2c-big-run-Final")




# # multiprocess environment
# n_cpu = multiprocessing.cpu_count()
# venv = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl0-v0') for _ in range(n_cpu)])
# env = CustomVecEnvWrapper(venv, (260, 520, 3), n_cpu)
#
# model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/", n_steps=stepsUpdate)
# model.learn(total_timesteps=int(1e2), tb_log_name="1MTimestepRun", callback=callback)
# env.close()
#
# venv = SubprocVecEnv([lambda: gym.make('gvgai-missilecommand-lvl0-v0') for _ in range(n_cpu)])
# env = CustomVecEnvWrapper(venv, (260, 520, 3), n_cpu)
# model.set_env(env)
# model.learn(total_timesteps=int(1e2), tb_log_name="1MTimestepRun_part2", callback=callback)
#
# model.save("models/a2c-boulderdash-lvl0-1M-Final")
# env.close()
