import sys

data_name = sys.argv[1]

meta_file = open('./raw_data/meta_{}.json'.format(data_name)).readlines()
out_file = open('./tmp/filtered_meta_{}.json'.format(data_name), 'w')

print("Filtering items with incomplete features...")

total_node_num = 0
feature_sets = [set() for _ in range(3)]

for eachline in meta_file:
    data = eval(eachline)
    if len(data['categories'][0]) >= 3 and 'brand' in data:
        cid1, cid2, cid3 = data['categories'][0][:3]
        bid = data['brand']
        features = [cid2, cid3, bid]
        for i in range(len(features)):
            feature_sets[i].add(features[i])
        out_file.write(eachline)
        total_node_num += 1

for i in range(len(features)):
    print(len(feature_sets[i]))
print('Total node num is {}'.format(total_node_num))

