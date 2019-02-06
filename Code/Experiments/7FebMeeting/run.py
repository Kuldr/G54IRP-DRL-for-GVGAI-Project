import sys
import gym
import gym_gvgai
import tensorflow as tf
import numpy as np

from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

# Custom CNN policy as per Deep Reinforcement Learning for General Video Game AI
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

            layer_1 = activ(conv(self.processed_obs, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
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


# python run.py STRING_OF_THE_ENVIRONMENT NUM_PROCESS TENSORBOARD_DIR TENSORBOARD_NAME SAVE_DIR NUM_TIMESTEPS
# python 0      1                         2           3               4                5        6
# python run.py "gvgai-boulderdash-lvl1-v0" 12 "tensorboard/a2c/" "Test1" "models/a2c/a2c-boulderdash-Lvl1-1K" 1000

# Get the variables from the command line
envString = sys.argv[1]
nProcesses = int(sys.argv[2])
logDir = sys.argv[3]
logName = sys.argv[4]
saveDir = sys.argv[5]
timesteps = int(sys.argv[6])

env = SubprocVecEnv([lambda: gym.make(envString) for i in range(nProcesses)])
model = A2C(CustomPolicy, env, verbose=1, tensorboard_log=logDir)
model.learn(total_timesteps=timesteps, tb_log_name=logName)
model.save(saveDir)
env.close()
