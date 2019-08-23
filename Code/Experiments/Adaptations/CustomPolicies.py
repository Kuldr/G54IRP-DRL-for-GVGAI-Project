import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

# Custom CNN policy as per Deep Reinforcement Learning for General Video Game AI
# Removes the Alpha channel
# Name Depth Kernel Stride
# C1 32 8 4
# C2 64 4 2
# C3 64 3 1
# FC1 256
class NatureCNN(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(NatureCNN, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            input = self.processed_obs

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

        total = 0
        for v in tf.trainable_variables():
            dims = v.get_shape().as_list()
            num  = int(np.prod(dims))
            total += num
            print('  %s \t\t Num: %d \t\t Shape %s ' % (v.name, num, dims))
        print('\nTotal number of params: %d' % total)

    def step(self, obs, state=None, mask=None, deterministic=False):
        action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})

class ONet(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(ONet, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            input = self.processed_obs

            with tf.variable_scope("CNNlayers"):
                with tf.variable_scope("CNNLayer1"):
                    layer_11 = activ(conv(input, 'c11', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_12 = activ(conv(layer_11, 'c12', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_13 = tf.nn.max_pool(layer_12, name="p13", ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("CNNLayer2"):
                    layer_21 = activ(conv(layer_13, 'c21', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_22 = activ(conv(layer_21, 'c22', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_23 = tf.nn.max_pool(layer_22, name="p23", ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("CNNLayer3"):
                    layer_31 = activ(conv(layer_23, 'c31', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_32 = activ(conv(layer_31, 'c32', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_33 = activ(conv(layer_32, 'c33', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_34 = tf.nn.max_pool(layer_33, name="p34", ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("CNNLayer4"):
                    layer_41 = activ(conv(layer_34, 'c41', n_filters=512, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_42 = activ(conv(layer_41, 'c42', n_filters=512, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_43 = activ(conv(layer_42, 'c43', n_filters=512, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_44 = tf.nn.max_pool(layer_43, name="p44", ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                with tf.variable_scope("CNNLayer5"):
                    layer_51 = activ(conv(layer_44, 'c51', n_filters=512, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_52 = activ(conv(layer_51, 'c52', n_filters=512, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_53 = activ(conv(layer_52, 'c53', n_filters=512, filter_size=3, stride=1, init_scale=np.sqrt(2), pad="SAME", **kwargs))
                    layer_54 = tf.nn.max_pool(layer_53, name="p54", ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope("DenseLayer6"):
                layer_61 = conv_to_fc(layer_54)
                layer_62 = activ(linear(layer_61, 'fc62', n_hidden=1024, init_scale=np.sqrt(2)))
                layer_63 = activ(linear(layer_62, 'fc63', n_hidden=256, init_scale=np.sqrt(2)))
                layer_64 = activ(linear(layer_63, 'fc64', n_hidden=64, init_scale=np.sqrt(2)))

            value_fn = tf.layers.dense(layer_64, 1, name='vf')

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(layer_64, layer_64, init_scale=0.01)

            total = 0
            for v in tf.trainable_variables():
                dims = v.get_shape().as_list()
                num  = int(np.prod(dims))
                total += num
                print('  %s \t\t Num: %d \t\t Shape %s ' % (v.name, num, dims))
            print('\nTotal number of params: %d' % total)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})
