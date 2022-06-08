import os
from collections import defaultdict
import json
import dgl
from tqdm import tqdm
import numpy as np
import torch
import networkx as nx
from utils import weighted_path_score
from torch.utils.data import Dataset

class REDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

def preprocess_data(in_file, out_file, word2id, ner2id, rel2id, type='train', fact_in_train=None, max_length=512):
    fact_in_train = fact_in_train if fact_in_train else set()
    data = []
    
    with open(in_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    for doc_id, doc in tqdm(enumerate(raw_data), total=len(raw_data)):

        title, sents, entities, labels = doc['title'], doc['sents'], doc['vertexSet'], doc.get('labels', [])
        num_sents = len(sents)
        num_labels = len(rel2id)
        num_entities = len(entities)

        lengths = [0]
        length = 0
        sent2pos = dict()
        for sent_id, sent in enumerate(sents):
            length += len(sent)
            lengths.append(length)
            sent2pos[sent_id] = (lengths[sent_id], lengths[sent_id + 1])

        for i in range(len(entities)):
            for j in range(len(entities[i])):
                sent_id = entities[i][j]['sent_id']
                base = lengths[sent_id]
                pos = entities[i][j]['pos']
                if pos[0] == pos[1]:
                    pos[1] += 1
                entities[i][j]['pos'] = (base + pos[0], base + pos[1])

        distance_mapping = np.zeros(max_length, dtype=np.int32)
        distance_mapping[1] = 1
        distance_mapping[2:] = 2
        distance_mapping[4:] = 3
        distance_mapping[8:] = 4
        distance_mapping[16:] = 5
        distance_mapping[32:] = 6
        distance_mapping[64:] = 7
        distance_mapping[128:] = 8
        distance_mapping[256:] = 9

        words = [word for sent in sents for word in sent]
        if len(words) > max_length:
            words = words[:max_length]

        word_ids = np.zeros(max_length, dtype=np.int32)
        ner_ids = np.zeros(max_length, dtype=np.int32)

        for i, word in enumerate(words):
            word_id = word2id.get(word.lower(), word2id['UNK'])
            word_ids[i] = word_id

        mention2sent = defaultdict(int)
        entity2mention = defaultdict(list)
        mention2pos = dict()
        mention_id = 0
        for entity_id, entity in enumerate(entities):
            overlapped_mention = set()
            for mention in entity:
                pos, ner, sent_id = mention['pos'], mention['type'], mention['sent_id']
                if pos in overlapped_mention:
                    continue
                ner_ids[pos[0]:pos[1]] = ner2id[ner]
                entity2mention[entity_id].append(mention_id)
                mention2sent[mention_id] = sent_id
                mention2pos[mention_id] = pos
                mention_id += 1
                overlapped_mention.add(pos)
                
        num_mentions = mention_id

        word_ids = torch.LongTensor(word_ids)
        ner_ids = torch.LongTensor(ner_ids)

        graph = build_graph(entity2mention, mention2sent, num_sents)
        node_type_seg = [0, num_entities, num_entities + num_mentions, num_entities + num_mentions + num_sents]
        nx_graph = dgl.to_networkx(dgl.to_homogeneous(graph))
        nx_graph = nx.Graph(nx_graph.to_undirected())
        for nx_sent_id in range(node_type_seg[2], node_type_seg[3] - 1):
            nx_graph.edges[nx_sent_id, nx_sent_id + 1]['weight'] = 4

        num_rels = 0
        new_labels = defaultdict(dict)
        for label in labels:
            head_entity_id, tail_entity_id, relation_name = label['h'], label['t'], label['r']
            rel_id = rel2id[relation_name]

            if (head_entity_id, tail_entity_id) in new_labels:
                rel_ids = new_labels[(head_entity_id, tail_entity_id)]['rel_ids']
                rel_ids[rel_id] = 1
            else:
                if type == 'train':
                    for head_mention in entities[head_entity_id]:
                        for tail_mention in entities[tail_entity_id]:
                            head_mention_name, tail_mention_name = head_mention['name'], tail_mention['name']
                            fact_in_train.add((head_mention_name, tail_mention_name, relation_name))
                    rel_ids = np.zeros(num_labels, dtype=np.int32)
                    rel_ids[rel_id] = 1
                    new_labels[(head_entity_id, tail_entity_id)]['rel_ids'] = rel_ids
                else:
                    in_train = False
                    for head_mention in entities[head_entity_id]:
                        for tail_mention in entities[tail_entity_id]:
                            head_mention_name, tail_mention_name = head_mention['name'], tail_mention['name']
                            if (head_mention_name, tail_mention_name, relation_name) in fact_in_train:
                                in_train = True
                                break
                        else:
                            continue
                    rel_ids = np.zeros(num_labels, dtype=np.int32)
                    rel_ids[rel_id] = 1
                    new_labels[(head_entity_id, tail_entity_id)]['rel_ids'] = rel_ids
                    new_labels[(head_entity_id, tail_entity_id)]['in_train'] = in_train
                num_rels += 1

        na_rel_ids = np.zeros(num_labels, dtype=np.int32)
        na_rel_ids[0] = 1
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j and (i, j) not in new_labels:
                    new_labels[(i, j)]['rel_ids'] = na_rel_ids
                    num_rels += 1

        in_trains = []
        subgraphs = []
        head_tail_entity_pairs = []
        dis_ids = np.zeros(num_rels, dtype=np.int32)
        labels = np.zeros((num_rels, num_labels), dtype=np.int32)
        for i, ((head_entity_id, tail_entity_id), rel_info) in enumerate(new_labels.items()):
            head_tail_entity_pairs.append([head_entity_id, tail_entity_id])
            rel_ids = rel_info['rel_ids']
            labels[i] = rel_ids
            in_train = rel_info.get('in_train', False)
            in_trains.append(in_train)
            cutoff = 4
            paths = [path for path in nx.all_simple_paths(nx_graph, head_entity_id, tail_entity_id, cutoff=cutoff) if weighted_path_score(nx_graph, path) <= cutoff]
            while not paths:
                cutoff += 4
                paths = [path for path in nx.all_simple_paths(nx_graph, head_entity_id, tail_entity_id, cutoff=cutoff) if weighted_path_score(nx_graph, path) <= cutoff]
            node_set = {node for path in paths for node in path}

            entity, mention, sentence = [], [], []
            for node in node_set:
                if node >= node_type_seg[2]:
                    sentence.append(node - node_type_seg[2])
                elif node >= node_type_seg[1]:
                    mention.append(node - node_type_seg[1])
                else:
                    entity.append(node)
            entity.sort()
            mention.sort()
            sentence.sort()

            subgraph_nodes = {'entity': entity, 'mention': mention, 'sentence': sentence}
            subgraph = graph.subgraph(subgraph_nodes)
            subgraph_head_entity_id = entity.index(head_entity_id)
            subgraph_tail_entity_id = entity.index(tail_entity_id)

            subgraph = dgl.add_nodes(subgraph, num=1, ntype='context')
            subgraph.add_edges(subgraph_head_entity_id, 0, etype=('entity', 'ec', 'context'))
            subgraph.add_edges(subgraph_tail_entity_id, 0, etype=('entity', 'ec', 'context'))
            subgraph.add_edges(0, subgraph_head_entity_id, etype=('context', 'ce', 'entity'))
            subgraph.add_edges(0, subgraph_tail_entity_id, etype=('context', 'ce', 'entity'))
            subgraph.add_edges(0, 0, etype=('context', 'cc', 'context'))

            subgraphs.append(subgraph)

            head_entity_pos, tail_entity_pos = entities[head_entity_id][0]['pos'], entities[tail_entity_id][0]['pos']
            if head_entity_pos[1] < tail_entity_pos[0]:
                abs_dis = tail_entity_pos[0] - head_entity_pos[1]
                abs_dis_id = distance_mapping[abs_dis]
                dis_id = 10 + abs_dis_id
            elif head_entity_pos[0] > tail_entity_pos[1]:
                abs_dis = head_entity_pos[0] - tail_entity_pos[1]
                abs_dis_id = distance_mapping[abs_dis]
                dis_id = 10 - abs_dis_id
            else:
                dis_id = 10
            dis_ids[i] = dis_id

        dis_ids = torch.LongTensor(dis_ids)
        labels = torch.LongTensor(labels)
        subgraphs = dgl.batch(subgraphs)
        sent_length = len(words)

        item = {
            'subgraphs': subgraphs,
            'labels': labels,
            'head_tail_entity_pairs': head_tail_entity_pairs,
            'dis_ids': dis_ids,
            'in_trains': in_trains,
            'title': title,
            'word_ids': word_ids,
            'sent_length': sent_length,
            'ner_ids': ner_ids,
            'sent2pos': sent2pos,
            'mention2pos': mention2pos,
            'entity2mention': entity2mention
        }
        data.append(item)

    torch.save(data, out_file)

    return fact_in_train

def build_graph(entity2mention, mention2sent, num_sents):
    num_nodes_dict = {'entity': len(entity2mention), 'mention': len(mention2sent), 'sentence': num_sents, 'context': 0}
    data_dict = defaultdict(list)

    data_dict[('context', 'cc', 'context')] = []
    data_dict[('context', 'ce', 'entity')] = []
    data_dict[('entity', 'ec', 'context')] = []

    for entity_id, mentions in entity2mention.items():
        data_dict[('entity', 'ee', 'entity')].append((entity_id, entity_id))
        for mention_id in mentions:
            data_dict[('mention', 'mm', 'mention')].append((mention_id, mention_id))
            data_dict[('mention', 'me', 'entity')].append((mention_id, entity_id))
            data_dict[('entity', 'em', 'mention')].append((entity_id, mention_id))

    for mention_id, sent_id in mention2sent.items():
        data_dict[('mention', 'ms', 'sentence')].append((mention_id, sent_id))
        data_dict[('sentence', 'sm', 'mention')].append((sent_id, mention_id))

    for i in range(num_sents - 1):
        data_dict[('sentence', 'ss', 'sentence')].append((i, i + 1))
        data_dict[('sentence', 'ss', 'sentence')].append((i + 1, i))

    for i in range(num_sents):
        data_dict[('sentence', 'ss', 'sentence')].append((i, i))

    graph = dgl.heterograph(data_dict, num_nodes_dict)

    return graph

if __name__ == '__main__':
    data_dir = 'data'
    rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), 'r'))
    ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), 'r'))

    train_in_file = os.path.join(data_dir, 'train_annotated.json')
    dev_in_file = os.path.join(data_dir, 'dev.json')
    test_in_file = os.path.join(data_dir, 'test.json')
    train_out_file = os.path.join(data_dir, 'prepro_data', 'train.pt')
    dev_out_file = os.path.join(data_dir, 'prepro_data', 'dev.pt')
    test_out_file = os.path.join(data_dir, 'prepro_data', 'test.pt')

    fact_in_train = preprocess_data(train_in_file, train_out_file, word2id, ner2id, rel2id, 'train')
    
    preprocess_data(dev_in_file, dev_out_file, word2id, ner2id, rel2id, 'dev', fact_in_train)

    preprocess_data(test_in_file, test_out_file, word2id, ner2id, rel2id, 'test', fact_in_train)
