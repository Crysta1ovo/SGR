import argparse
import random
import numpy as np
import torch
import dgl

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--vec_path', type=str, default='data/vec.npy')
    parser.add_argument('--train_path', type=str, default='data/prepro_data/train.pt')
    parser.add_argument('--dev_path', type=str, default='data/prepro_data/dev.pt')
    parser.add_argument('--test_path', type=str, default='data/prepro_data/test.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-6)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_norm', type=float, default=1.0)

    parser.add_argument('--entity_type_embed_size', type=int, default=20)
    parser.add_argument('--coref_embed_size', type=int, default=20)
    parser.add_argument('--dis_embed_size', type=int, default=20)
    parser.add_argument('--num_entity_types', type=int, default=7)
    parser.add_argument('--num_dis_types', type=int, default=20)
    parser.add_argument('--max_num_entities', type=int, default=42)
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--rgcn_hidden_size', type=int, default=768)
    parser.add_argument('--fc_hidden_size', type=int, default=1024)
    parser.add_argument('--num_rels', type=int, default=97)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    return args
    
def weighted_path_score(G, path):
   edges = zip(path, path[1:])
   return sum(G.edges[u, v].get('weight', 1) for u, v in edges)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    word_ids = torch.stack([item['word_ids'] for item in batch])
    ner_ids = torch.stack([item['ner_ids'] for item in batch])
    labels = torch.cat([item['labels'] for item in batch], dim=0)
    dis_ids = torch.cat([item['dis_ids'] for item in batch], dim=0)
    head_tail_entity_pairs = [item['head_tail_entity_pairs'] for item in batch]
    in_trains = [in_train for item in batch for in_train in item['in_trains']]
    title = [item['title'] for item in batch]
    sent_lengths = [item['sent_length'] for item in batch]
    sent2pos = [item['sent2pos'] for item in batch]
    mention2pos = [item['mention2pos'] for item in batch]
    entity2mention = [item['entity2mention'] for item in batch]
    subgraphs = [item['subgraphs'] for item in batch]
    entity_indices, mention_indices, sentence_indices = [], [], []
    for subgraph in subgraphs:
        entity_indices.append(subgraph.ndata[dgl.NID]['entity'].tolist())
        mention_indices.append(subgraph.ndata[dgl.NID]['mention'].tolist())
        sentence_indices.append(subgraph.ndata[dgl.NID]['sentence'].tolist())
    node_indices = (entity_indices, mention_indices, sentence_indices)
    subgraphs = dgl.batch(subgraphs)

    return {
        'subgraphs': subgraphs,
        'node_indices': node_indices,
        'labels': labels,
        'head_tail_entity_pairs': head_tail_entity_pairs,
        'dis_ids': dis_ids,
        'in_trains': in_trains,
        'title': title,
        'word_ids': word_ids,
        'sent_lengths': sent_lengths,
        'ner_ids': ner_ids,
        'sent2pos': sent2pos,
        'mention2pos': mention2pos,
        'entity2mention': entity2mention,
    }
