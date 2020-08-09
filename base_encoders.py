import tensorflow as tf
from tf_euler.python import layers

from tf_euler.python import aggregators
from tf_euler.python import encoders
from tf_euler.python import euler_ops


class BaseSageEncoder(layers.Layer):
  @staticmethod
  def create_aggregators(dim, num_layers, aggregator, **kwargs):
    new_aggregators = []
    aggregator_class = aggregators.get(aggregator)
    for layer in range(num_layers):
      activation = tf.nn.relu if layer < num_layers - 1 else None
      new_aggregators.append(
          aggregator_class(dim, activation=activation, **kwargs))
    return new_aggregators

  def __init__(self, metapath, fanouts, dim,
               aggregator='mean', concat=False, shared_aggregators=None,
               feature_idx=-1, feature_dim=0, max_id=-1,
               use_feature=None, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False,
               shared_node_encoder=None, use_residual=False, *args, **kwargs):
    super(BaseSageEncoder, self).__init__(**kwargs)
    if len(metapath) != len(fanouts):
      raise ValueError('Len of metapath must be the same as fanouts.')
    if use_feature is not None or use_id is not None:
      tf.logging.warning('use_feature is deprecated '
                         'and would not have any effect.')

    self.metapath = metapath
    self.fanouts = fanouts
    self.num_layers = len(metapath)
    self.concat = concat

    if shared_node_encoder:
      self._node_encoder = shared_node_encoder
    else:
      self._node_encoder = encoders.ShallowEncoder(
          feature_idx=feature_idx, feature_dim=feature_dim,
          max_id=max_id if use_id else -1,
          sparse_feature_idx=sparse_feature_idx,
          sparse_feature_max_id=sparse_feature_max_id,
          embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding)

    layer0_dim = self._node_encoder.output_dim
    self.dims = [layer0_dim] + [dim] * self.num_layers

    if shared_aggregators is not None:
      self.aggregators = shared_aggregators
    else:
      self.aggregators = self.create_aggregators(
          dim, self.num_layers, aggregator, concat=concat)

    self.dim = dim
    self._max_id = max_id

  def sample_fanout(self, inputs):
    raise NotImplementedError

  def neigh_processing(self, neigh_hidden, neigh_shape):
    return NotImplementedError

  def aggregation(self, aggregator, node_hidden, neigh_hidden):
    raise NotImplementedError

  def node_encoder(self, inputs):
    return self._node_encoder(inputs)

  def call(self, inputs):
    samples = self.sample_fanout(inputs)
    hidden = [self.node_encoder(sample) for sample in samples]
    for layer in range(self.num_layers):
      aggregator = self.aggregators[layer]
      next_hidden = []
      for hop in range(self.num_layers - layer):
        neigh_shape = [-1, self.fanouts[hop], self.dims[layer]]
        neigh_hidden = self.neigh_processing(hidden[hop+1], neigh_shape)
        h = self.aggregation(aggregator, hidden[hop], neigh_hidden)
        # next_hidden.append(tf.nn.l2_normalize(h, axis=-1))
        next_hidden.append(h)
      hidden = next_hidden
    output_shape = inputs.shape.concatenate(self.dims[-1])
    output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
    hidden = tf.reshape(hidden[0], output_shape)
    return hidden
