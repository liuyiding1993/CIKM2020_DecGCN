import sys
from collections import defaultdict


def main():
  data_name = sys.argv[1]
  f = open("./tmp/filtered_meta_{}.json".format(data_name), 'r').readlines() 
  sim_edges = defaultdict(int)
  rel_edges = defaultdict(int)

  print("Filtering items with not sub/comp edges...")

  all_asin = set()
  for eachline in f:
    each_data = eval(eachline)
    asin = each_data['asin']
    all_asin.add(asin)

  for eachline in f:
    each_data = eval(eachline)
    asin = each_data['asin']
    if 'related' in each_data:
      related_data = each_data['related']

      if 'also_viewed' in related_data:
        for rid in related_data['also_viewed']:
          if rid not in all_asin: continue
          u, v = str(asin), str(rid)
          if u > v: u, v = v, u
          edge = (u, v)
          sim_edges[edge] += 1

      if 'also_bought' in related_data:
        for rid in related_data['also_bought']:
          if rid not in all_asin: continue
          u, v = str(asin), str(rid)
          if u > v: u, v = v, u
          edge = (u, v)
          rel_edges[edge] += 1
    
  fout_sim = open("./tmp/{}_sim.edges".format(data_name), 'w') 
  fout_rel = open("./tmp/{}_cor.edges".format(data_name), 'w') 

  max_sim_weight = 0
  max_rel_weight = 0
  all_nodes = set()
  for (u, v), w in sim_edges.items():
    fout_sim.write('\t'.join([u, v, str(w)]) + '\n')
    max_sim_weight = max(max_sim_weight, w)
    all_nodes.add(u)
    all_nodes.add(v)
  for (u, v), w in rel_edges.items():
    fout_rel.write('\t'.join([u, v, str(w)]) + '\n')
    max_rel_weight = max(max_rel_weight, w)
    all_nodes.add(u)
    all_nodes.add(v)
  
  print("max_sim_weight: {}; max_rel_weight: {}; node num: {}".format(
         max_sim_weight, max_rel_weight, len(all_nodes)))
    

if __name__ == "__main__":
  main()

