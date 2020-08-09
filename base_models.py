import tensorflow as tf
from tf_euler.python import layers
from unsupervised_model import UnsupervisedModel, ModelOutput


class BaseGraphSage(UnsupervisedModel):
  def __init__(self, node_type, edge_type, max_id,
               encoder_class=None, metapath=None,
               *args, **kwargs):
    super(BaseGraphSage, self).__init__(
        node_type, edge_type, max_id, *args, **kwargs)
    if encoder_class and metapath:
        for i, mp in enumerate(metapath):
          print("Metapath #{}: {}".format(i, mp))
        self.encoder_class = encoder_class
        self._sim_target_encoder = self.new_encoder(metapath[0], *args, **kwargs)
        self._sim_context_encoder = self.new_encoder(metapath[1], *args, **kwargs)
        self._cor_target_encoder = self.new_encoder(metapath[2], *args, **kwargs)
        self._cor_context_encoder = self.new_encoder(metapath[3], *args, **kwargs)

  def new_encoder(self, metapath, *args, **kwargs):
    return self.encoder_class(metapath, *args, **kwargs)

  def embedding_dim(self):
      return self._sim_target_encoder.dim


class BaseGraphSage_SE(BaseGraphSage):
  def __init__(self, node_type, edge_type, max_id, *args, **kwargs):
    super(BaseGraphSage_SE, self).__init__(node_type, edge_type, max_id, *args, **kwargs)
    dim = self.embedding_dim()
    self.sim_to_cor = [layers.Dense(dim, activation=tf.nn.relu, use_bias=True) for _ in range(2)]
    self.cor_to_sim = [layers.Dense(dim, activation=tf.nn.relu, use_bias=True) for _ in range(2)]

  def translate(self, embedding, kernels):
    for kernel in kernels:
        embedding = kernel(embedding)
    return embedding

  def postencoding(self, src, embedding, edge_type):
    aux_encoder = None
    trans_kernel = to_aux_kernel = None

    if edge_type == self.eval_sim_edge_type:
      aux_encoder = self.cor_target_encoder
      trans_kernel = self.cor_to_sim
      to_aux_kernel = self.sim_to_cor
    elif edge_type == self.eval_cor_edge_type:
      aux_encoder = self.sim_target_encoder
      trans_kernel = self.sim_to_cor
      to_aux_kernel = self.cor_to_sim

    aux_embedding = aux_encoder(src)
    forward_residual_embedding = self.translate(aux_embedding, trans_kernel)
    cycle_residual_embedding = self.translate(self.translate(embedding, to_aux_kernel) + aux_embedding, trans_kernel)
    embedding = embedding + cycle_residual_embedding + forward_residual_embedding
    return embedding

