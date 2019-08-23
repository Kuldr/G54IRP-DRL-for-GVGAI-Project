import gym
import gym_gvgai
import multiprocessing
import numpy as np

from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines import A2C

from CustomPolicies import ONet, NatureCNN
from EnvWrapper import EnvWrapper

RENDER_TO_SCREEN = False
stepsUpdate = 5 # 1 to render each frame | Otherwise not really sure why you want it larger
callbacks = 0
folderName = "Test/"
runName = "Aliens-lvl1"

def callback(locals, _):
    global callbacks
    callbacks += 1
    if RENDER_TO_SCREEN:
        locals["self"].env.render()
    # Saves the model every 1000 calls
    if callbacks % 1000 == 0:
        locals['self'].save("models/" + folderName + runName + "-" + str(callbacks))
    return True # Returns true as false ends the training

n = 1
list = [lambda: gym.make('gvgai-aliens-lvl1-v0') for _ in range(n)]

# multiprocess environment
n_cpu = multiprocessing.cpu_count()
venv = SubprocVecEnv(list)
venv = EnvWrapper(venv, (128, 128, 3))
model = A2C(ONet, venv, verbose=1, tensorboard_log="tensorboard/"+folderName+"/", n_steps=stepsUpdate)
model.learn(total_timesteps=int(1e6), tb_log_name=runName, callback=callback)
venv.close()
model.save("models/" + folderName + runName + "-Final")
