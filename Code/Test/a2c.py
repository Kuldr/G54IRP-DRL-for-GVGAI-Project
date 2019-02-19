import gym
import gym_gvgai
import multiprocessing
import tensorflow as tf
import numpy as np
from PIL import Image

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines import A2C

class CustomVecEnvWrapper(VecEnvWrapper):
    # Not happy how actions and observations are reshaped

    # TODO:
    #   Make action transforming its own functions

    # I don't think rendering to screen works
    def __init__(self, venv, desiredShape):
        self.venv = venv
        self.desiredShape = desiredShape
        (self.y, self.x, self.c) = desiredShape
        self.b = len(self.venv.remotes)

        # Manually get the dtype
        # self.venv.remotes[0].send(('get_spaces', None))
        # obsSpace, _ = self.venv.remotes[0].recv()
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

        for remote, action in zip(self.venv.remotes, actions):
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
        self.venv.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.venv.remotes]
        self.venv.waiting = False
        obs, rews, dones, infos = zip(*results)
        returnList = []
        for frame in obs:
            returnList.append(self.transfromFrame(frame))
        return np.stack(returnList), np.stack(rews), np.stack(dones), infos

        # observations, rewards, dones, infos = self.venv.step_wait()
        # return self.transformBatch(observations), rewards, dones, infos

    def reset(self):
        # This doesn't work if each environment has a different size
        for remote in self.venv.remotes:
            remote.send(('reset', None))
        resetFrames = [remote.recv() for remote in self.venv.remotes]
        returnList = []
        for frame in resetFrames:
            returnList.append(self.transfromFrame(frame))
        return np.stack(returnList)
        #
        # obs = self.venv.reset()
        # return self.transformBatch(obs)

    def close(self):
        self.venv.close()

    def transfromFrame(self, frame):
        frame = frame[:,:,:3]
        # Convert to PIL Image and resize before converting back and adding to new array
        frameIm = Image.fromarray(frame)
        frameIm = frameIm.resize((self.x,self.y))
        frame = np.asarray(frameIm)
        return frame

    def transformBatch(self, batchObs):
        # # Slice off the alpha channel
        # batchObs = batchObs[:,:,:,:3]

        # Resize transformation
        resizedBatchObs = np.empty((self.b, self.y, self.x, self.c), dtype=np.uint8) # Create output array
        for i, frame in enumerate(batchObs[:]):
            frame = self.transfromFrame(frame)
            resizedBatchObs[i] = frame

        # Name output and return
        observation = resizedBatchObs
        return observation

# Custom CNN policy as per Deep Reinforcement Learning for General Video Game AI
# Removes the Alpha channel
# Name Depth Kernel Stride
# C1 32 8 4
# C2 64 4 2
# C3 64 3 1
# FC1 256
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            input = self.processed_obs

            # # Transform input layer
            # with tf.name_scope("Transform"):
            #     sliceA = tf.slice(self.processed_obs, [0,0,0,0], [-1,-1,-1,3], "SliceA")
            #     resize = tf.image.resize_image_with_pad(sliceA, 300, 300)#, ResizeMethod.NEAREST_NEIGHBOR)
            #     input = resize

            layer_1 = activ(conv(input, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
            layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
            layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
            layer_3 = conv_to_fc(layer_3)
            extracted_features = activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))

            value_fn = tf.layers.dense(extracted_features, 1, name='vf')

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(extracted_features, extracted_features, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None):
        action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})

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
