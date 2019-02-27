import numpy as np
import gym
from PIL import Image

from stable_baselines.common.vec_env import VecEnvWrapper

from modelHelperFunctions import transformFrame

class CustomVecEnvWrapper(VecEnvWrapper):
    # Not happy how actions and observations are reshaped

    # TODO:
    #   Make action transforming its own functions

    # I don't think rendering to screen works
    def __init__(self, venv, desiredShape):
        self.vecenv = venv.venv
        self.desiredShape = desiredShape
        (self.y, self.x, self.c) = desiredShape
        self.b = len(self.vecenv.remotes)

        # Manually get the dtype
        # self.vecenv.remotes[0].send(('get_spaces', None))
        # obsSpace, _ = self.vecenv.remotes[0].recv()
        # dtype = obsSpace.dtype
        # print(dtype)
        # print(np.iinfo(dtype).max)
        # print(np.iinfo(dtype).min)

        # Create the new shapes for actions and observations
        observation_space = gym.spaces.Box(low=0, high=255, shape=desiredShape, dtype=np.uint8)
        actionSpace = gym.spaces.Discrete(6)

        VecEnvWrapper.__init__(self, venv, observation_space=observation_space, action_space=actionSpace)

    def step_async(self, actions):
        # actions is a list of ints for each action for each env

        for remote, action in zip(self.vecenv.remotes, actions):
            # remote.send(('get_attr', 'action_space'))
            remote.send(('get_spaces', None))
            _, actionSpace = remote.recv()
            # print("Action Space: " + str(actionSpace.n))
            # print("Action Chose: " + str(action))
            if not action >= actionSpace.n:
                remote.send(('step', action))
            else:
                # print("Action Changed to 0")
                remote.send(('step', 0)) # Send the default ACTION_NIL code
        self.vecenv.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.vecenv.remotes]
        self.vecenv.waiting = False
        obs, rews, dones, infos = zip(*results)
        returnList = []
        for frame in obs:
            returnList.append(transformFrame(frame, x=self.x, y=self.y))
        return np.stack(returnList), np.stack(rews), np.stack(dones), infos

    def reset(self):
        # This doesn't work if each environment has a different size
        for remote in self.vecenv.remotes:
            remote.send(('reset', None))
        resetFrames = [remote.recv() for remote in self.vecenv.remotes]
        returnList = []
        for frame in resetFrames:
            returnList.append(transformFrame(frame, x=self.x, y=self.y))
        return np.stack(returnList)

    def close(self):
        self.vecenv.close()
