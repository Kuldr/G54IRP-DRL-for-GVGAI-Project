import gym
import gym_gvgai
import multiprocessing

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

RENDER_TO_SCREEN = False
stepsUpdate = 5 # 1 to render each frame | Otherwise not really sure why you want it larger

callbacks = 0

def callback(locals, _):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global callbacks
    callbacks += 1
    if RENDER_TO_SCREEN:
        locals["self"].env.render()
    # Saves the model every 1000 calls
    if callbacks % 1000 == 0:
        locals['self'].save("models/a2c-boulderdash-lvl0-1M-" + str(callbacks))
    return True # Returns true as false ends the training

# multiprocess environment
n_cpu = multiprocessing.cpu_count()
env = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl0-v0') for _ in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/", n_steps=stepsUpdate)
model.learn(total_timesteps=int(1e6), tb_log_name="1MTimestepRun", callback=callback)
model.save("models/a2c-boulderdash-lvl0-1M-Final")
env.close()
