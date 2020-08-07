import json
import sys


def assign_num_id(u, node_to_num_id):
  if u not in node_to_num_id:
    node_to_num_id[u] = len(node_to_num_id)
  return node_to_num_id[u]


def init_node(u, num_id):
  return {
    "node_id": u, 
    "num_id": num_id,
    "node_weight": 1.0,
    "sim_neighbor": {},
    "sim_edge": [],
    "cor_neighbor": {},
    "cor_edge": []
  }


def add_neighbor_and_edge(u, v, edge_type, node_data):
  current_node_data = node_data[u]
  current_node_data[edge_type + "_edge"].append([u, v, 1.0])
  current_node_data[edge_type + "_neighbor"][v] = 1.0


def main():
  '''
  data format:
  {
  "node_id": str,
  "num_id": int,
  "node_weight": float,
  "sim_neighbor": {"nb_id": "nb_weight"},
  "cor_neighbor": {"nb_id": "nb_weight"},
  "sim_edge":[{
    "src_id": int,
    "dst_id": int,
    "weight": float,
  }]
  "cor_edge":[{
    "src_id": int,
    "dst_id": int,
    "weight": float,
  }]
  }
  '''
  print("Converting node data to json format...")

  sim_filename = "{}_sim.edges".format(sys.argv[1])
  cor_filename = "{}_cor.edges".format(sys.argv[1])

  sim_edges_file = open("./tmp/" + sim_filename, 'r') 
  cor_edges_file = open("./tmp/" + cor_filename, 'r') 
  
  node_to_num_id = {}
  node_data = {}

  for eachline in sim_edges_file:
    u, v, w = eachline.strip('\n').split('\t') 
    uid = assign_num_id(u, node_to_num_id) 
    vid = assign_num_id(v, node_to_num_id) 

    if u not in node_data:
      node_data[u] = init_node(u, uid) 
    if v not in node_data:
      node_data[v] = init_node(v, vid)

    add_neighbor_and_edge(u, v, "sim", node_data) 
    add_neighbor_and_edge(v, u, "sim", node_data) 
  
  for eachline in cor_edges_file:
    u, v, w = eachline.strip('\n').split('\t') 
    uid = assign_num_id(u, node_to_num_id) 
    vid = assign_num_id(v, node_to_num_id) 

    if u not in node_data:
      node_data[u] = init_node(u, uid) 
    if v not in node_data:
      node_data[v] = init_node(v, vid)

    add_neighbor_and_edge(u, v, "cor", node_data) 
    add_neighbor_and_edge(v, u, "cor", node_data) 

  print("Total node num: {}".format(len(node_data)))
  json.dump(node_data, open("./data/{}_node_data.json".format(sys.argv[1]), 'w'))

  id_dict_file = open('./tmp/{}_id_dict.txt'.format(sys.argv[1]), 'w')
  for node_id in node_data:
    num_id = node_data[node_id]['num_id']
    id_dict_file.write("{}\t{}\n".format(node_id, num_id))


if __name__ == '__main__':
  main()

