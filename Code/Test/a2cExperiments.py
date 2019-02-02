import gym
import gym_gvgai
import multiprocessing
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

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

# Run 1M timestep run for each lvl
env = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl0-v0') for _ in range(12)])
model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/")
model.learn(total_timesteps=int(1e6), tb_log_name="1MLvl0")
model.save("models/a2c-boulderdash-lvl0-1M")
env.close()
del model

env = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl1-v0') for _ in range(12)])
model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/")
model.learn(total_timesteps=int(1e6), tb_log_name="1MLvl1")
model.save("models/a2c-boulderdash-lvl1-1M")
env.close()
del model

env = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl2-v0') for _ in range(12)])
model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/")
model.learn(total_timesteps=int(1e6), tb_log_name="1MLvl2")
model.save("models/a2c-boulderdash-lvl2-1M")
env.close()
del model

env = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl3-v0') for _ in range(12)])
model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/")
model.learn(total_timesteps=int(1e6), tb_log_name="1MLvl3")
model.save("models/a2c-boulderdash-lvl3-1M")
env.close()
del model

env = SubprocVecEnv([lambda: gym.make('gvgai-boulderdash-lvl4-v0') for _ in range(12)])
model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/")
model.learn(total_timesteps=int(1e6), tb_log_name="1MLvl4")
model.save("models/a2c-boulderdash-lvl4-1M")
env.close()
del model

list = [lambda: gym.make('gvgai-boulderdash-lvl0-v0') for _ in range(3)] + [lambda: gym.make('gvgai-boulderdash-lvl1-v0') for _ in range(3)] + [lambda: gym.make('gvgai-boulderdash-lvl2-v0') for _ in range(3)] + [lambda: gym.make('gvgai-boulderdash-lvl3-v0') for _ in range(3)] + [lambda: gym.make('gvgai-boulderdash-lvl4-v0') for _ in range(3)]
env = SubprocVecEnv(list)
model = A2C(CustomPolicy, env, verbose=1, tensorboard_log="tensorboard/a2cBoulderdash/")
model.learn(total_timesteps=int(1e6), tb_log_name="1MAll3times")
model.save("models/a2c-boulderdash-AllLevels-1M")
env.close()
del model
