import tensorflow as tf
from base_encoders import BaseSageEncoder

from tf_euler.python import layers
from tf_euler.python import euler_ops

from samplers import vanilla_sample_fanout
from samplers import sample_sim_cor_fanout


class SageEncoder(BaseSageEncoder):
  def __init__(self, *args, **kwargs):
    super(SageEncoder, self).__init__(*args, **kwargs)

  def sample_fanout(self, inputs):
    return vanilla_sample_fanout(
        inputs, self.metapath, self.fanouts, default_node=self._max_id + 1)[0]

  def neigh_processing(self, neigh_hidden, neigh_shape):
    return tf.reshape(neigh_hidden, neigh_shape)

  def aggregation(self, aggregator, node_hidden, neigh_hidden):
    return aggregator((node_hidden, neigh_hidden))


class ST_SageEncoder(BaseSageEncoder):
  def __init__(self, target_type, *args, **kwargs):
    super(ST_SageEncoder, self).__init__(*args, **kwargs)
    self.target_type = target_type

  def sample_fanout(self, inputs):
    return sample_sim_cor_fanout(
      inputs, self.metapath, self.fanouts, default_node=self._max_id + 1)[0]

  def neigh_processing(self, neigh_hidden, neigh_shape):
    neigh_sim_hidden, neigh_cor_hidden = tf.split(neigh_hidden, num_or_size_splits=2, axis=0)
    neigh_sim_hidden = tf.reshape(neigh_sim_hidden, neigh_shape)
    neigh_cor_hidden = tf.reshape(neigh_cor_hidden, neigh_shape)
    return neigh_sim_hidden, neigh_cor_hidden

  def coattention(self, D, Q):
    L = tf.matmul(tf.transpose(D, perm=[0, 2, 1]), Q)
    AQ = tf.nn.softmax(L, axis=-1)
    AD = tf.nn.softmax(tf.transpose(L, perm=[0, 2, 1]), axis=-1)
    CQ = tf.matmul(D, AQ)
    CD = tf.matmul(tf.concat([Q, CQ], axis=1), AD)
    return tf.transpose(tf.concat([CD, CQ], axis=1), perm=[0, 2, 1]), L 

  def aggregation(self, aggregator, node_hidden, neigh_hidden):
    D = Q = None
    neigh_sim_hidden, neigh_cor_hidden = neigh_hidden
    if self.target_type == 'sim':
      D = tf.transpose(neigh_sim_hidden, perm=[0, 2, 1])
      Q = tf.transpose(neigh_cor_hidden, perm=[0, 2, 1])
    elif self.target_type == 'cor':
      D = tf.transpose(neigh_cor_hidden, perm=[0, 2, 1])
      Q = tf.transpose(neigh_sim_hidden, perm=[0, 2, 1])
    neigh_hidden, self.L = self.coattention(D, Q)
    return aggregator((node_hidden, neigh_hidden))
