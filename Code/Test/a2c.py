import gym
import gym_gvgai
import multiprocessing
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines import A2C

from PIL import Image

class CustomVecEnvWrapper(VecEnvWrapper):
    # TODO:
    #   Make the transformations their own functions so its easier to edit
    #   Add in scaling / padding / both
    #   Test on training with 2 games at once | different actions sizes may matter
    # IDEAS:
    #   Do I reshape the action space to be a constant size
    #       What happens if I try games that don't have the same action space size
    #       Might need to / be able to get batch size for nenvs
    #   Set the Observation dType - are they all the same?

    def __init__(self, venv, desiredShape, nEnvs):
        self.venv = venv
        self.desiredShape = desiredShape
        (self.y, self.x, self.c) = desiredShape
        self.b = nEnvs # Would be nicer to get this from venv

        # Could set the observation dtype?
        # Can you manually get low and high from dtype?
        observation_space = gym.spaces.Box(low=0, high=255, shape=desiredShape, dtype=np.uint8)

        VecEnvWrapper.__init__(self, venv, observation_space=observation_space, action_space=venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        return self.transform(observations), rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        return self.transform(obs)

    def close(self):
        self.venv.close()

    def transform(self, batchObs):
        # Slice off the alpha channel
        batchObs = batchObs[:,:,:,:3]

        # Resize transformation
        resizedBatchObs = np.empty((self.b, self.y, self.x, self.c), dtype=np.uint8) # Create output array
        for i, frame in enumerate(batchObs[:]):
            # Convert to PIL Image and resize before converting back and adding to new array
            frameIm = Image.fromarray(frame)
            frameIm = frameIm.resize((self.x,self.y))
            frame = np.asarray(frameIm)
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
        locals['self'].save("models/a2c-boulderdash-lvl0-1M-" + str(callbacks))
    return True # Returns true as false ends the training

# multiprocess environment
n_cpu = multiprocessing.cpu_count()
venv = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl0-v0') for _ in range(n_cpu)])
env = CustomVecEnvWrapper(venv, (260, 520, 3), n_cpu)

model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/", n_steps=stepsUpdate)
model.learn(total_timesteps=int(1e2), tb_log_name="1MTimestepRun", callback=callback)
env.close()

venv = SubprocVecEnv([lambda: gym.make('gvgai-missilecommand-lvl0-v0') for _ in range(n_cpu)])
env = CustomVecEnvWrapper(venv, (260, 520, 3), n_cpu)
model.set_env(env)
model.learn(total_timesteps=int(1e2), tb_log_name="1MTimestepRun_part2", callback=callback)

model.save("models/a2c-boulderdash-lvl0-1M-Final")
env.close()
