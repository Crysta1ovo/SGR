import os
import numpy as np
import torch
import torch.nn as nn
from model import Model
from tqdm import tqdm
from utils import set_seed, collate_fn, parse_args
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
from preprocess import REDataset
from torch.utils.data import DataLoader
from test import evaluation

logging.basicConfig(
    level=logging.INFO,
    filename='./logs/docre.log',
    filemode='w'
)
logger = logging.getLogger(__name__)
    
if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    word2vec = np.load(args.vec_path)
    word2vec[1] = word2vec[2:].mean(axis=0)

    train_dataset = REDataset(args.train_path)
    dev_dataset = REDataset(args.dev_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)

    model = Model(word2vec, args)
    
    start_epoch = 1
    best_f1 = 0.
    total_num_rels = 12275
    num_training_steps = args.num_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    num_warmup_steps = num_training_steps * args.warmup_ratio
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    with tqdm(total=num_training_steps) as pbar:
        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            model.train()
            for i, batch in enumerate(train_dataloader):
                subgraphs, node_indices, labels, head_tail_entity_pairs, dis_ids, in_trains, title, word_ids, sent_lengths, ner_ids, sent2pos, mention2pos, entity2mention = batch.values()
                subgraphs, dis_ids, labels, word_ids, ner_ids = subgraphs.to(device), dis_ids.to(device), labels.to(device), word_ids.to(device), ner_ids.to(device)
                labels = labels.float()
                outputs = model(subgraphs, node_indices, head_tail_entity_pairs, dis_ids, word_ids, sent_lengths, ner_ids, sent2pos, mention2pos, entity2mention)
                loss = criterion(outputs, labels)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    pbar.set_postfix_str(f'{loss.item():.4f}')
                    pbar.update(1)

            if epoch == 1 or epoch >= args.num_epochs - 5:
                model.eval()
                f1, ign_f1, epoch_loss, threshold = evaluation(model, dev_dataloader, criterion, total_num_rels, device)  
                if f1 > best_f1:
                    best_f1 = f1
                    path = os.path.join('checkpoints', f'sgr.pt')
                    torch.save({'f1': f1, 'threshold': threshold, 'model': model.state_dict()}, path)

                print(f'Epoch: {epoch:02}')
                print(f'\tValid Loss: {epoch_loss / len(dev_dataloader):.3f} | Valid F1: {f1*100:.2f}% | Valid Ign F1: {ign_f1*100:.2f}% | Threshold: {threshold}')
                logger.info(f'Epoch: {epoch:02}')
                logger.info(f'\tValid Loss: {epoch_loss / len(dev_dataloader):.3f} | Valid F1: {f1*100:.2f}% | Valid Ign F1: {ign_f1*100:.2f}% | Threshold: {threshold}')
