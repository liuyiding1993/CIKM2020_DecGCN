import json
import sys


def process_edges():
    train_sim_edge_num, test_sim_edge_num = 0, 0
    train_cor_edge_num, test_cor_edge_num = 0, 0
    out_file = open('./euler_data/{}.edges'.format(data_name), 'w')
    for node in node_data:
        if len(node_data[node]['sim_edge']) > 0:
            for u, v, w in node_data[node]['sim_edge'][:-1]:
                uid = str(node_data[u]['num_id'])
                vid = str(node_data[v]['num_id'])
                etype = '0'
                out_file.write(';'.join([uid, vid, etype, '1.0']) + '\n')
                train_sim_edge_num += 1

            u, v, w = node_data[node]['sim_edge'][-1]
            uid = str(node_data[u]['num_id'])
            vid = str(node_data[v]['num_id'])
            etype = '2'
            out_file.write(';'.join([uid, vid, etype, '1.0']) + '\n')
            test_sim_edge_num += 1
        if len(node_data[node]['cor_edge']) > 0:
            for u, v, w in node_data[node]['cor_edge'][:-1]:
                uid = str(node_data[u]['num_id'])
                vid = str(node_data[v]['num_id'])
                etype = '1'
                out_file.write(';'.join([uid, vid, etype, '1.0']) + '\n')
                train_cor_edge_num += 1
            u, v, w = node_data[node]['cor_edge'][-1]
            uid = str(node_data[u]['num_id'])
            vid = str(node_data[v]['num_id'])
            etype = '3'
            out_file.write(';'.join([uid, vid, etype, '1.0']) + '\n')
            test_cor_edge_num += 1
    print("Train -- sim edge num: {}; cor edge num: {}".format(train_sim_edge_num, train_cor_edge_num), file=log_file)
    print("Test -- sim edge num: {}; cor edge num: {}".format(test_sim_edge_num, test_cor_edge_num), file=log_file)


def id_mapping(s, id_dict):
    if s not in id_dict:
        id_dict[s] = len(id_dict)
    return id_dict[s]


def process_nodes():
    node_num = 0
    meta_file = open('./tmp/filtered_meta_{}.json'.format(data_name)).readlines()
    out_file = open('./euler_data/{}.points'.format(data_name), 'w')
    cid1_dict, cid2_dict, cid3_dict = {}, {}, {}
    bid_dict = {}
    out_buffer = {}
    for eachline in meta_file:
        data = eval(eachline)
        u = data['asin']
        if u in node_data:
            c1, c2, c3 = data['categories'][0][:3]
            brand = data['brand']
            uid = node_data[u]['num_id']
            cid1, cid2, cid3 = id_mapping(c1, cid1_dict), id_mapping(c2, cid2_dict), \
                               id_mapping(c3, cid3_dict)
            bid = id_mapping(brand, bid_dict)
            node_num += 1
            out_buffer[uid] = ';'.join(map(str, [uid, cid2, cid3, bid])) + '\n'
    for uid in sorted(out_buffer.keys()):
        out_file.write(out_buffer[uid])

    print("node num: {}".format(node_num), file=log_file)
    print("#cid2: {}; #cid2: {}; #bid: {}".format(len(cid2_dict), len(cid3_dict), len(bid_dict)), file=log_file)


if __name__ == '__main__':
    print("Converting node data to euler format...")

    data_name = sys.argv[1]
    log_filename = "./data_stats/{}_data_stats.log".format(data_name)
    log_file = open(log_filename, 'w')
    print(data_name, file=log_file)

    node_data = json.load(open('./data/{}_node_data.json'.format(data_name), 'r'))
    process_edges()
    process_nodes()

    print("Finished... stats saved in {}".format(log_filename))

