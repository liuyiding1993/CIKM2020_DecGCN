import tensorflow as tf
from tf_euler.python import euler_ops


vanilla_sample_fanout = euler_ops.sample_fanout


def sample_sim_cor_fanout(nodes, edge_types, counts, default_node=-1):
  neighbors_list = [tf.reshape(nodes, [-1])]
  for hop_edge_types, count in zip(edge_types, counts):
    sim_edge_type = cor_edge_type = None
    if len(hop_edge_types) == 2:
      sim_edge_type, cor_edge_type = hop_edge_types
    elif len(hop_edge_types) == 1:
      sim_edge_type = cor_edge_type = hop_edge_types[0]
    sim_neighbors, _, _ = euler_ops.sample_neighbor(
      neighbors_list[-1], [sim_edge_type], count, default_node=default_node)
    cor_neighbors, _, _ = euler_ops.sample_neighbor(
      neighbors_list[-1], [cor_edge_type], count, default_node=default_node)
    sim_neighbors = tf.reshape(sim_neighbors, [-1])
    cor_neighbors = tf.reshape(cor_neighbors, [-1])
    neighbors = tf.concat([sim_neighbors, cor_neighbors], axis=-1)
    neighbors_list.append(neighbors)
  return [neighbors_list]


def sample_patterned_metapaths(nodes, patterns, count_per_path, default_node=-1):
  all_neighbors = []
  for meta_pattern in patterns:
    pattern_neighbors = nodes
    last_neighbors = tf.reshape(nodes, [-1])
    counts = [int(count_per_path // len(meta_pattern))] + [1] * (len(meta_pattern) - 1)
    for hop_edge_types, count in zip(meta_pattern, counts):
      neighbors, _, _ = euler_ops.sample_neighbor(
          last_neighbors, [hop_edge_types], count, default_node=default_node)
      last_neighbors = tf.reshape(neighbors, [-1])
      neighbors = tf.reshape(neighbors, [-1, counts[0]])
      pattern_neighbors = tf.concat([pattern_neighbors, neighbors], axis=-1)
    all_neighbors.append(pattern_neighbors)
  all_nodes = [nodes] * len(patterns)
  return all_nodes, all_neighbors

