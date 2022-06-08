import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import os
from preprocess import REDataset
from utils import collate_fn, set_seed, parse_args
from model import Model

def evaluation(model, dev_dataloader, criterion, total_num_rels, device):
    epoch_loss = 0.
    items = []
    for batch in tqdm(dev_dataloader, total=len(dev_dataloader)):
        subgraphs, node_indices, labels, head_tail_entity_pairs, dis_ids, in_trains, title, word_ids, sent_lengths, ner_ids, sent2pos, mention2pos, entity2mention = batch.values()
        subgraphs, dis_ids, labels, word_ids, ner_ids = subgraphs.to(device), dis_ids.to(device), labels.to(device), word_ids.to(device), ner_ids.to(device)
        labels = labels.float()
        with torch.no_grad():
            outputs = model(subgraphs, node_indices, head_tail_entity_pairs, dis_ids, word_ids, sent_lengths, ner_ids, sent2pos, mention2pos, entity2mention)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        probs = torch.sigmoid(outputs)
        labels_no_na = labels[:, 1:].cpu().numpy()
        probs_no_na = probs[:, 1:].cpu().numpy()
        for i in range(labels_no_na.shape[0]):
            for j in range(labels_no_na.shape[1]):
                items.append((labels_no_na[i][j], probs_no_na[i][j], in_trains[i]))

    items.sort(key=lambda x: x[1], reverse=True)
    precision, recall = [], []
    correct = 0
    for idx, item in enumerate(items):
        correct += item[0]
        precision.append(float(correct) / (idx + 1))
        recall.append(float(correct) / total_num_rels)
    precision = np.asarray(precision, dtype='float32')
    recall = np.asarray(recall, dtype='float32')
    f1_arr = (2 * precision * recall / (precision + recall + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    threshold = items[f1_pos][1]

    precision, recall = [], []
    correct = correct_in_train = 0
    for idx, item in enumerate(items):
        correct += item[0]
        if item[0] and item[2]:
            correct_in_train += 1
        precision.append(float(correct - correct_in_train) / max((idx + 1 - correct_in_train), 1))
        recall.append(float(correct) / total_num_rels)
    precision = np.asarray(precision, dtype='float32')
    recall = np.asarray(recall, dtype='float32')
    ign_f1_arr = (2 * precision * recall / (precision + recall + 1e-20))
    ign_f1 = ign_f1_arr.max()

    return f1, ign_f1, epoch_loss, threshold

def test(model, test_dataloader, threshold, id2rel, device):
    data = []

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        subgraphs, node_indices, labels, head_tail_entity_pairs, dis_ids, in_trains, title, word_ids, sent_lengths, ner_ids, sent2pos, mention2pos, entity2mention = batch.values()
        subgraphs, dis_ids, word_ids, ner_ids = subgraphs.to(device), dis_ids.to(device), word_ids.to(device), ner_ids.to(device)
        with torch.no_grad():
            outputs = model(subgraphs, node_indices, head_tail_entity_pairs, dis_ids, word_ids, sent_lengths, ner_ids, sent2pos, mention2pos, entity2mention)
        titles = []
        for i, head_tail_entity_pair in enumerate(head_tail_entity_pairs):
            titles += [title[i]] * len(head_tail_entity_pair)
        head_tail_entity_pairs = [ht for head_tail_entity_pair in head_tail_entity_pairs for ht in head_tail_entity_pair]
        probs = torch.sigmoid(outputs)
        subgraph_ids, relation_ids = torch.where(probs[:, 1:] >= threshold)
        for subgraph_id, relation_id in zip(subgraph_ids.tolist(), relation_ids.tolist()):
            item = {
                'title': titles[subgraph_id],
                'h_idx': head_tail_entity_pairs[subgraph_id][0],
                't_idx': head_tail_entity_pairs[subgraph_id][1],
                'r': id2rel[relation_id + 1],
            }
            data.append(item)
                    
    json.dump(data, open('result.json', 'w'))

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parse_args()
    set_seed(args.seed)
    word2vec = np.load(args.vec_path)
    word2vec[1] = word2vec[2:].mean(axis=0)

    test_path = os.path.join(args.test_path)
    test_dataset = REDataset(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    rel2id = json.load(open(os.path.join(args.data_dir, 'rel2id.json'), "r"))
    id2rel = {v: k for k, v in rel2id.items()}
    model = Model(word2vec, args)
    path = os.path.join('checkpoints', 'sgr.pt')
    checkpoint = torch.load(path)
    print(checkpoint['f1'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    test(model, test_dataloader, checkpoint['threshold'], id2rel, device)
