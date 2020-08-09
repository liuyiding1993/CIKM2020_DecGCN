import tensorflow as tf

from base_models import BaseGraphSage
from base_models import BaseGraphSage_SE
from encoders import ST_SageEncoder
from encoders import SageEncoder 


class VanillaGraphSage(BaseGraphSage):
  def __init__(self, node_type, edge_type, max_id, metapath, *args, **kwargs):
    sim_metapath = [[p] for p in metapath[0]]
    cor_metapath = [[p] for p in metapath[1]]
    encoder_class = SageEncoder
    metapath = [sim_metapath, sim_metapath, cor_metapath, cor_metapath]
    super(VanillaGraphSage, self).__init__(node_type, edge_type, max_id, encoder_class, metapath, *args, **kwargs)


class GraphSage_ST(BaseGraphSage):
  def __init__(self, node_type, edge_type, max_id, metapath, *args, **kwargs):
    super(GraphSage_ST, self).__init__(node_type, edge_type, max_id, *args, **kwargs)
    metapath = [metapath] * 4
    encoder_type = ['sim', 'sim', 'cor', 'cor']
    for i, mp in enumerate(metapath):
      print("Metapath #{}: {}".format(i, mp))
    self.encoder_class = ST_SageEncoder 
    self._sim_target_encoder = self.new_encoder(encoder_type[0], metapath[0], *args, **kwargs)
    self._sim_context_encoder = self.new_encoder(encoder_type[1], metapath[1], *args, **kwargs)
    self._cor_target_encoder = self.new_encoder(encoder_type[2], metapath[2], *args, **kwargs)
    self._cor_context_encoder = self.new_encoder(encoder_type[3], metapath[3], *args, **kwargs)


class GraphSage_SE(BaseGraphSage_SE, VanillaGraphSage):
  def __init__(self, node_type, edge_type, max_id, metapath, *args, **kwargs):
    super(GraphSage_SE, self).__init__(node_type, edge_type, max_id, metapath, *args, **kwargs)


class DecGCN(BaseGraphSage_SE, GraphSage_ST):
  def __init__(self, node_type, edge_type, max_id, metapath,  *args, **kwargs):
    super(DecGCN, self).__init__(node_type, edge_type, max_id, metapath, *args, **kwargs)

