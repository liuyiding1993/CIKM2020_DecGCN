import sys
from collections import defaultdict


def main():
  print("Filtering nodes with edge num threshold = {}".format(threshold))

  sim_filename = "{}_sim.edges".format(sys.argv[1])
  cor_filename = "{}_cor.edges".format(sys.argv[1])

  sim_edges = open("./tmp/" + sim_filename, 'r').readlines() 
  cor_edges = open("./tmp/" + cor_filename, 'r').readlines() 
  invalid_nodes = set()

  filter_round = 0
  while 1:
    filter_round += 1
    print("Round {}".format(filter_round))

    node_sim_score = defaultdict(int)
    node_cor_score = defaultdict(int)
    valid_nodes = set()

    for eachline in sim_edges:
      u, v, _ = eachline.strip('\n').split('\t')
      if u in invalid_nodes or v in invalid_nodes:
        continue
      node_sim_score[u] += 1
      node_sim_score[v] += 1
      node_cor_score[u] += 0
      node_cor_score[v] += 0
      valid_nodes.add(u)
      valid_nodes.add(v)
  
    for eachline in cor_edges:
      u, v, _ = eachline.strip('\n').split('\t')
      if u in invalid_nodes or v in invalid_nodes:
        continue
      node_cor_score[u] += 1
      node_cor_score[v] += 1
      node_sim_score[u] += 0
      node_sim_score[v] += 0
      valid_nodes.add(u)
      valid_nodes.add(v)
  
    print("Num of valid nodes: {}".format(len(valid_nodes)))

    stop_sign = True 
    for u, s in node_sim_score.items():
      if s < threshold: 
        invalid_nodes.add(u) 
        stop_sign = False 
    for u, s in node_cor_score.items():
      if s < threshold: 
        invalid_nodes.add(u) 
        stop_sign = False
    print("Num of filtered nodes: {}".format(len(invalid_nodes)))
    if stop_sign: break
    print("")

  filtered_sim_file = open("./tmp/filtered_{}".format(sim_filename), 'w')
  filtered_cor_file = open("./tmp/filtered_{}".format(cor_filename), 'w')

  for eachline in sim_edges:
    u, v, _ = eachline.strip('\n').split('\t')
    if u in invalid_nodes or v in invalid_nodes:
      continue
    filtered_sim_file.write(eachline)
  
  for eachline in cor_edges:
    u, v, _ = eachline.strip('\n').split('\t')
    if u in invalid_nodes or v in invalid_nodes:
      continue
    filtered_cor_file.write(eachline)


if __name__ == "__main__":
  threshold = 0 
  main()

