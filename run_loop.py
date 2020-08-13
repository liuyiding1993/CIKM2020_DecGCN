# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from itertools import product

from model import GraphSage_ST 
from model import GraphSage_SE 
from model import DecGCN

from tf_euler.python import models
from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python import optimizers
from tf_euler.python.utils import context as utils_context
from tf_euler.python.utils import hooks as utils_hooks
from euler.python import service

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def define_network_embedding_flags():
  tf.flags.DEFINE_enum('mode', 'train',
                       ['train', 'evaluate', 'save_embedding'], 'Run mode.')

  tf.flags.DEFINE_string('data_dir', 'euler_data/Beauty', 'Local Euler graph data.')

  tf.flags.DEFINE_integer('max_id', 114792, 'Max node id.')
  tf.flags.DEFINE_list('sparse_feature_idx', [0, 1, 2], 'Sparse feature index')
  tf.flags.DEFINE_list('sparse_feature_max_id', [11, 45, 11179], 'Sparse feature max id')

  tf.flags.DEFINE_integer('train_node_type', 0, 'Node type of training set.')
  tf.flags.DEFINE_integer('all_node_type', euler_ops.ALL_NODE_TYPE,
                          'Node type of the whole graph.')
  tf.flags.DEFINE_list('train_edge_type', [0, 1], 'Edge type of training set.')
  tf.flags.DEFINE_list('eval_edge_type', [2, 3], 'Edge type of training set.')
  tf.flags.DEFINE_list('all_edge_type', [0, 1, 2, 3],
                       'Edge type of the whole graph.')

  tf.flags.DEFINE_integer('feature_idx', -1, 'Feature index.')
  tf.flags.DEFINE_integer('feature_dim', 0, 'Feature dimension.')
  tf.flags.DEFINE_integer('label_idx', -1, 'Label index.')
  tf.flags.DEFINE_integer('label_dim', 0, 'Label dimension.')

  tf.flags.DEFINE_integer('num_classes', None, 'Number of classes.')
  tf.flags.DEFINE_list('id_file', [], 'Files containing ids to evaluate.')

  tf.flags.DEFINE_boolean('sigmoid_loss', True, 'Whether to use sigmoid loss.')
  tf.flags.DEFINE_boolean('xent_loss', True, 'Whether to use xent loss.')

  tf.flags.DEFINE_integer('dim', 128, 'Dimension of embedding.')
  tf.flags.DEFINE_integer('embedding_dim', 16, 'Dimension of embedding.')
  tf.flags.DEFINE_integer('num_negs', 5, 'Number of negative samplings.')

  tf.flags.DEFINE_string('model', 'DecGCN', 'Embedding model.')
  tf.flags.DEFINE_list('fanouts', [5, 5], 'GCN fanouts.')
  tf.flags.DEFINE_enum('aggregator', 'mean',
                       ['gcn', 'mean', 'meanpool', 'maxpool', 'attention'],
                       'Sage aggregator.')
  tf.flags.DEFINE_boolean('concat', True, 'Sage aggregator concat.')
  tf.flags.DEFINE_boolean('use_residual', False, 'Whether use skip connection.')

  tf.flags.DEFINE_string('model_dir', 'ckpt', 'Model checkpoint.')
  tf.flags.DEFINE_integer('batch_size', 512, 'Mini-batch size.')
  tf.flags.DEFINE_string('optimizer', 'adam', 'Optimizer to use.')
  tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
  tf.flags.DEFINE_integer('num_epochs', 20, 'Number of epochs for training.')
  tf.flags.DEFINE_integer('log_steps', 20, 'Number of steps to print log.')

  tf.flags.DEFINE_list('ps_hosts', [], 'Parameter servers.')
  tf.flags.DEFINE_list('worker_hosts', [], 'Training workers.')
  tf.flags.DEFINE_string('job_name', '', 'Cluster role.')
  tf.flags.DEFINE_integer('task_index', 0, 'Task index.')

  tf.flags.DEFINE_string('euler_zk_addr', '127.0.0.1:2181',
                         'Euler ZK registration service.')
  tf.flags.DEFINE_string('euler_zk_path', '/tf_euler',
                         'Euler ZK registration node.')


def run_train(model, flags_obj, master, is_chief):
  utils_context.training = True

  batch_size = flags_obj.batch_size // model.batch_size_ratio
  source = euler_ops.sample_node(
      count=batch_size, node_type=flags_obj.train_node_type)
  source.set_shape([batch_size])

  sim_outputs, cor_outputs = model(source)
  _, sim_loss, metric_name, sim_metric = sim_outputs
  _, cor_loss, ___________, cor_metric = cor_outputs
  loss = sim_loss + cor_loss

  optimizer_class = optimizers.get(flags_obj.optimizer)
  optimizer = optimizer_class(flags_obj.learning_rate)
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.minimize(loss, global_step=global_step)

  hooks = []

  tensor_to_log = {'step': global_step, 'loss': loss, 
                   'sim_loss': sim_loss, 'cor_loss': cor_loss, 
                   'sim_metric': sim_metric, 'cor_metric': cor_metric}
  hooks.append(
      tf.train.LoggingTensorHook(
          tensor_to_log, every_n_iter=flags_obj.log_steps))

  num_steps = int((flags_obj.max_id + 1) // flags_obj.batch_size *
                   flags_obj.num_epochs)
  hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

  extra_param_name = '_'.join(map(str, flags_obj.fanouts)) 
  output_dir = ckpt_dir = '{}/{}/{}_{}_{}/'.format(
      flags_obj.model_dir, flags_obj.model, 
      extra_param_name,
      flags_obj.dim, flags_obj.embedding_dim)
  print("output dir: {}".format(output_dir))

  if len(flags_obj.worker_hosts) == 0 or flags_obj.task_index == 1:
    hooks.append(
        tf.train.ProfilerHook(save_secs=180, output_dir=output_dir))
  if len(flags_obj.worker_hosts):
    hooks.append(utils_hooks.SyncExitHook(len(flags_obj.worker_hosts)))
  if hasattr(model, 'make_session_run_hook'):
    hooks.append(model.make_session_run_hook())

  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=ckpt_dir,
      log_step_count_steps=None,
      hooks=hooks,
      config=config) as sess:
    while not sess.should_stop():
      sess.run(train_op)


def run_evaluate(model, flags_obj, master, is_chief):
  tf.random.set_random_seed(20191223)
  utils_context.training = False

  ckpt_dir = '{}/{}/{}_{}_{}/'.format(
      flags_obj.model_dir, flags_obj.model, 
      '_'.join(map(str, flags_obj.fanouts)), 
      flags_obj.dim, flags_obj.embedding_dim)
  id_file = ckpt_dir + "/id.txt"

  dataset = tf.data.TextLineDataset(id_file)
  if master:
    dataset = dataset.shard(len(flags_obj.worker_hosts), flags_obj.task_index)
  dataset = dataset.map(
      lambda id_str: tf.string_to_number(id_str, out_type=tf.int64))

  dataset = dataset.batch(flags_obj.batch_size)
  source = dataset.make_one_shot_iterator().get_next()

  sim_outputs, cor_outputs = model(source)
  _, _, metric_name, sim_metric = sim_outputs
  _, _, ___________, cor_metric = cor_outputs

  tf.train.get_or_create_global_step()

  hooks = []
  num_steps = int((flags_obj.max_id + 1) // flags_obj.batch_size *
                 flags_obj.num_epochs)
  hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

  if master:
    hooks.append(utils_hooks.SyncExitHook(len(flags_obj.worker_hosts)))

  batch_sim_metric_val, batch_cor_metric_val = [], []
  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=ckpt_dir,
      save_checkpoint_secs=None,
      log_step_count_steps=None,
      hooks=hooks,
      config=config) as sess:
    while not sess.should_stop():
      sim_metric_val, cor_metric_val = sess.run(
        [sim_metric, cor_metric])
      batch_sim_metric_val.append(sim_metric_val)
      batch_cor_metric_val.append(cor_metric_val)

  sim_metric_val = np.mean(batch_sim_metric_val)
  cor_metric_val = np.mean(batch_cor_metric_val)
  print('sim_{}: {}'.format(metric_name, sim_metric_val))
  print('cor_{}: {}'.format(metric_name, cor_metric_val))


def run_save_embedding(model, flags_obj, master, is_chief):
  tf.random.set_random_seed(20191223)
  utils_context.training = False

  dataset = tf.data.Dataset.range(flags_obj.max_id + 1)
  if master:
    dataset = dataset.shard(len(flags_obj.worker_hosts), flags_obj.task_index)

  dataset = dataset.batch(flags_obj.batch_size)
  source = dataset.make_one_shot_iterator().get_next()

  sim_outputs, cor_outputs = model(source)
  sim_embedding, _, _, sim_metric = sim_outputs
  cor_embedding, _, _, cor_metric = cor_outputs

  tf.train.get_or_create_global_step()
  hooks = []
  if master:
    hooks.append(utils_hooks.SyncExitHook(len(flags_obj.worker_hosts)))

  extra_param_name = '_'.join(map(str, flags_obj.fanouts)) 
  model_dir = '{}/{}/{}_{}_{}/'.format(
      flags_obj.model_dir, flags_obj.model, 
      extra_param_name,
      flags_obj.dim, flags_obj.embedding_dim)

  ids = []
  sim_embedding_vals = []
  cor_embedding_vals = []
  sim_weight_vals = []
  cor_weight_vals = []
  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=model_dir,
      save_checkpoint_secs=None,
      log_step_count_steps=None,
      hooks=hooks,
      config=config) as sess:
    while not sess.should_stop():
      id_, sim_embedding_val, cor_embedding_val = sess.run([
          source, sim_embedding, cor_embedding])
      ids.append(id_)
      sim_embedding_val, sim_weight_val = sim_embedding_val[0], sim_embedding_val[1]
      cor_embedding_val, cor_weight_val = cor_embedding_val[0], cor_embedding_val[1]
      sim_embedding_vals.append(sim_embedding_val)
      cor_embedding_vals.append(cor_embedding_val)
      sim_weight_vals.append(sim_weight_val)
      cor_weight_vals.append(cor_weight_val)

  id_ = np.concatenate(ids)
  sim_embedding_val = np.concatenate(sim_embedding_vals)
  cor_embedding_val = np.concatenate(cor_embedding_vals)
  sim_weight_val = np.concatenate(sim_weight_vals)
  cor_weight_val = np.concatenate(cor_weight_vals)

  if master:
    sim_embedding_filename = 'sim_embedding_{}.npy'.format(flags_obj.task_index)
    cor_embedding_filename = 'cor_embedding_{}.npy'.format(flags_obj.task_index)
    sim_weight_filename = 'sim_weight_{}.npy'.format(flags_obj.task_index)
    cor_weight_filename = 'cor_weight_{}.npy'.format(flags_obj.task_index)
    id_filename = 'id_{}.txt'.format(flags_obj.task_index)
  else:
    sim_embedding_filename = 'sim_embedding.npy'
    cor_embedding_filename = 'cor_embedding.npy'
    sim_weight_filename = 'sim_weight.npy'
    cor_weight_filename = 'cor_weight.npy'
    id_filename = 'id.txt'
  sim_embedding_filename = model_dir + '/' + sim_embedding_filename
  cor_embedding_filename = model_dir + '/' + cor_embedding_filename
  sim_weight_filename = model_dir + '/' + sim_weight_filename
  cor_weight_filename = model_dir + '/' + cor_weight_filename
  id_filename = model_dir + '/' + id_filename

  with tf.gfile.GFile(sim_embedding_filename, 'w') as sim_embedding_file:
    np.save(sim_embedding_file, sim_embedding_val)
  with tf.gfile.GFile(cor_embedding_filename, 'w') as cor_embedding_file:
    np.save(cor_embedding_file, cor_embedding_val)

  with tf.gfile.GFile(sim_weight_filename, 'w') as sim_weight_file:
    np.save(sim_weight_file, sim_weight_val)
  with tf.gfile.GFile(cor_weight_filename, 'w') as cor_weight_file:
    np.save(cor_weight_file, cor_weight_val)

  with tf.gfile.GFile(id_filename, 'w') as id_file:
    id_file.write('\n'.join(map(str, id_)))


def run_network_embedding(flags_obj, master, is_chief):
  fanouts = map(int, flags_obj.fanouts)
  sparse_feature_idx = map(int, flags_obj.sparse_feature_idx)
  sparse_feature_max_id = map(int, flags_obj.sparse_feature_max_id)
  
  if flags_obj.mode == 'evaluate':
    eval_edge_type = flags_obj.eval_edge_type
  elif flags_obj.mode == 'save_embedding':
    eval_edge_type = flags_obj.eval_edge_type
  else:
    eval_edge_type = flags_obj.train_edge_type

  metapath = [map(int, flags_obj.train_edge_type)] * len(fanouts)

  if flags_obj.model == 'DecGCN/SE':
    model = GraphSage_ST(
        node_type=flags_obj.train_node_type,
        edge_type=flags_obj.train_edge_type,
        eval_edge_type = eval_edge_type,
        max_id=flags_obj.max_id,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        metapath=metapath,
        fanouts=fanouts,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        embedding_dim=flags_obj.embedding_dim,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id)

  if flags_obj.model == 'DecGCN/ST':
    metapath = [[int(l)] * len(fanouts) for l in flags_obj.train_edge_type]
    model = GraphSage_SE(
        node_type=flags_obj.train_node_type,
        edge_type=flags_obj.train_edge_type,
        eval_edge_type = eval_edge_type,
        max_id=flags_obj.max_id,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        metapath=metapath,
        fanouts=fanouts,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        embedding_dim=flags_obj.embedding_dim,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id)

  elif flags_obj.model == 'DecGCN':
    model = DecGCN(
        node_type=flags_obj.train_node_type,
        edge_type=flags_obj.train_edge_type,
        eval_edge_type = eval_edge_type,
        max_id=flags_obj.max_id,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        metapath=metapath,
        fanouts=fanouts,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        embedding_dim=flags_obj.embedding_dim,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id)

  if flags_obj.mode == 'train':
    run_train(model, flags_obj, master, is_chief)
  elif flags_obj.mode == 'evaluate':
    run_evaluate(model, flags_obj, master, is_chief)
  elif flags_obj.mode == 'save_embedding':
    run_save_embedding(model, flags_obj, master, is_chief)


def run_local(flags_obj, run):
  if not euler_ops.initialize_embedded_graph(flags_obj.data_dir):
    raise RuntimeError('Failed to initialize graph.')

  run(flags_obj, master='', is_chief=True)


def run_distributed(flags_obj, run):
  cluster = tf.train.ClusterSpec({
      'ps': flags_obj.ps_hosts,
      'worker': flags_obj.worker_hosts
  })
  server = tf.train.Server(
      cluster, job_name=flags_obj.job_name, task_index=flags_obj.task_index)

  if flags_obj.job_name == 'ps':
    server.join()
  elif flags_obj.job_name == 'worker':
    if not euler_ops.initialize_shared_graph(
        directory=flags_obj.data_dir,
        zk_addr=flags_obj.euler_zk_addr,
        zk_path=flags_obj.euler_zk_path,
        shard_idx=flags_obj.task_index,
        shard_num=len(flags_obj.worker_hosts),
        global_sampler_type='node'):
      raise RuntimeError('Failed to initialize graph.')

    with tf.device(
        tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % flags_obj.task_index,
            cluster=cluster)):
      run(flags_obj, server.target, flags_obj.task_index == 0)
  else:
    raise ValueError('Unsupport role: {}'.format(flags_obj.job_name))


def main(_):
  flags_obj = tf.flags.FLAGS
  if flags_obj.worker_hosts:
    run_distributed(flags_obj, run_network_embedding)
  else:
    run_local(flags_obj, run_network_embedding)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_network_embedding_flags()
  tf.app.run(main)
