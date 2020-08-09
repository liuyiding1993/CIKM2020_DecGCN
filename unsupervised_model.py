import collections
import tensorflow as tf

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python import metrics

ModelOutput = collections.namedtuple(
    'ModelOutput', ['embedding', 'loss', 'metric_name', 'metric'])


class Model(layers.Layer):
  """
  """

  def __init__(self, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.batch_size_ratio = 1


class UnsupervisedModel(Model):
  """
  Base model for unsupervised network embedding model.
  """

  def __init__(self,
               node_type,
               edge_type,
               max_id,
               eval_edge_type,
               num_negs=5,
               xent_loss=False,
               **kwargs):
    super(UnsupervisedModel, self).__init__(**kwargs)
    self.node_type = node_type

    self.edge_type = edge_type
    self.eval_edge_type = eval_edge_type

    print("Train edge type:{}; Eval edge type: {}".format(edge_type, eval_edge_type))

    self.max_id = max_id
    self.num_negs = num_negs
    self.xent_loss = xent_loss

    self.sim_edge_type = [edge_type[0]]
    self.cor_edge_type = [edge_type[1]]

    self.eval_sim_edge_type = [eval_edge_type[0]]
    self.eval_cor_edge_type = [eval_edge_type[1]]

  def sim_target_encoder(self, inputs):
    return self._sim_target_encoder(inputs)

  def sim_context_encoder(self, inputs):
    return self._sim_context_encoder(inputs)

  def cor_target_encoder(self, inputs):
    return self._cor_target_encoder(inputs)

  def cor_context_encoder(self, inputs):
    return self._cor_context_encoder(inputs)

  def _mrr(self, aff, aff_neg):
    aff_all = tf.concat([aff_neg, aff], axis=2)
    size = tf.shape(aff_all)[2]
    _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
    _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
    return tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, :, -1] + 1)))

  def decoder(self, embedding, embedding_pos, embedding_negs):
    logits = tf.matmul(embedding, embedding_pos, transpose_b=True)
    neg_logits = tf.matmul(embedding, embedding_negs, transpose_b=True)
    mrr = self._mrr(logits, neg_logits)
    if self.xent_loss:
      true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.ones_like(logits), logits=logits)
      negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(neg_logits), logits=neg_logits)
      loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
    else:
      neg_cost = tf.reduce_logsumexp(neg_logits, axis=2, keepdims=True)
      loss = -tf.reduce_sum(logits - neg_cost)
    return loss, mrr

  def to_sample(self, inputs, edge_type):
    batch_size = tf.size(inputs)
    src = tf.expand_dims(inputs, -1)
    pos = euler_ops.sample_neighbor(inputs, edge_type, 1,
                                    self.max_id + 1)[0]
    negs = euler_ops.sample_node(batch_size * self.num_negs, self.node_type)
    negs = tf.reshape(negs, [batch_size, self.num_negs])
    return src, pos, negs

  def postencoding(self, src, embedding, edge_type):
    return embedding

  def model_output(self, sampled_data, edge_type):
    src, pos, negs = sampled_data
    target_encoder = context_encoder = None
    if edge_type == self.eval_sim_edge_type:
      target_encoder = self.sim_target_encoder
      context_encoder = self.sim_context_encoder
    elif edge_type == self.eval_cor_edge_type:
      target_encoder = self.cor_target_encoder
      context_encoder = self.cor_context_encoder

    embedding = target_encoder(src)
    embedding = self.postencoding(src, embedding, edge_type)

    weight = context_encoder(src)
    embedding_pos = context_encoder(pos)
    embedding_negs = context_encoder(negs)
    loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
    return ModelOutput(
      embedding=[tf.squeeze(embedding), tf.squeeze(weight)],
      loss=loss, metric_name='mrr', metric=mrr)

  def call(self, inputs):
    sim_samples = self.to_sample(inputs, self.eval_sim_edge_type)
    cor_samples = self.to_sample(inputs, self.eval_cor_edge_type)
    sim_outputs = self.model_output(sim_samples, self.eval_sim_edge_type)
    cor_outputs = self.model_output(cor_samples, self.eval_cor_edge_type)
    return sim_outputs, cor_outputs
