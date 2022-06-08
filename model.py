import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, weights, args):
        super(Model, self).__init__()
        self.weights = weights
        self.vocab_size = weights.shape[0]
        self.embed_size = weights.shape[1] 
        self.entity_type_embed_size = args.entity_type_embed_size
        self.num_entity_types = args.num_entity_types
        self.num_dis_types = args.num_dis_types
        self.coref_embed_size = args.coref_embed_size
        self.max_num_entities = args.max_num_entities
        self.dis_embed_size = args.dis_embed_size
        self.lstm_hidden_size = args.lstm_hidden_size
        self.lstm_num_layers = args.lstm_num_layers
        self.rgcn_hidden_size = args.rgcn_hidden_size
        self.dropout = args.dropout
        self.num_rels = args.num_rels

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.weights), freeze=True, padding_idx=0)
        self.fc_hidden_size = args.fc_hidden_size
        self.rel_names = ['me', 'em', 'ms', 'sm', 'ce', 'ec', 'cc', 'mm', 'ee', 'ss']

        self.embed_size += self.entity_type_embed_size
        self.entity_type_embedding = nn.Embedding(self.num_entity_types, self.entity_type_embed_size, padding_idx=0)
        self.dis_embedding = nn.Embedding(self.num_dis_types, self.dis_embed_size, padding_idx=10)

        self.encoder = LSTM(self.embed_size, self.lstm_hidden_size, self.lstm_num_layers, self.dropout)
        self.gnn = HeteroClassifier(self.lstm_hidden_size * 2, self.rgcn_hidden_size, self.rel_names, self.dropout)

        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2 + self.rgcn_hidden_size * 2 + self.dis_embed_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_hidden_size, self.num_rels),
        )

        self.Wc = nn.Linear(self.lstm_hidden_size * 4, self.lstm_hidden_size * 2)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, subgraphs, node_indices, head_tail_entity_pairs, dis_ids, word_ids, sent_lengths, ner_ids, sent2pos, mention2pos, entity2mention):
        embedding = torch.cat([self.word_embedding(word_ids), self.entity_type_embedding(ner_ids)], dim=-1)
        encoder_outputs = self.encoder(embedding, sent_lengths)

        sentence_features = [torch.stack([torch.max(encoder_outputs[i][pos[0]:pos[1]], dim=0)[0] for pos in item.values()]) for i, item in enumerate(sent2pos)]
        mention_features = [torch.stack([torch.max(encoder_outputs[i][pos[0]:pos[1]], dim=0)[0] for pos in item.values()]) for i, item in enumerate(mention2pos)]
        entity_features = [torch.stack([torch.max(mention_features[i][mention_ids], dim=0)[0] for mention_ids in item.values()]) for i, item in enumerate(entity2mention)]

        entity_indices, mention_indices, sentence_indices = node_indices
        subgraph_sent_feat = torch.cat([sentence_feature[sentence_indice] for sentence_feature, sentence_indice in zip(sentence_features, sentence_indices)])
        subgraph_men_feat = torch.cat([mention_feature[mention_indice] for mention_feature, mention_indice in zip(mention_features, mention_indices)])
        subgraph_ent_feat = torch.cat([entity_feature[entity_indice] for entity_feature, entity_indice in zip(entity_features, entity_indices)])

        head_entity_feature, tail_entity_feature = [], []
        for i, head_tail_entity_pair in enumerate(head_tail_entity_pairs):
            tail_entity_feature.append(entity_features[i][[h for h, t in head_tail_entity_pair]])
            head_entity_feature.append(entity_features[i][[t for h, t in head_tail_entity_pair]])
        head_entity_feature = torch.cat(head_entity_feature)
        tail_entity_feature = torch.cat(tail_entity_feature)

        subgraphs.nodes['sentence'].data['feat'] = subgraph_sent_feat
        subgraphs.nodes['mention'].data['feat'] = subgraph_men_feat
        subgraphs.nodes['entity'].data['feat'] = subgraph_ent_feat
        subgraphs.nodes['context'].data['feat'] = torch.max(torch.stack([head_entity_feature, tail_entity_feature]), dim=0)[0]

        entity_feature = self.Wc(torch.cat([head_entity_feature, tail_entity_feature], dim=-1))

        document_feature, context_feature = self.gnn(subgraphs)
        
        dis_feature = self.dis_embedding(dis_ids)

        output = self.fc(torch.cat([entity_feature, document_feature, context_feature, dis_feature], dim=-1))
        
        return output

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, dropout):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: self.dropout(self.activation(v)) for k, v in h.items()}
        h = self.conv2(graph, h)

        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, rel_names, dropout):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names, dropout)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
   
        with g.local_scope():
            g.ndata['h'] = h
            document_feature = dgl.max_nodes(g, 'h', ntype='sentence')
            context_feature = h['context']

            return document_feature, context_feature

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, embedding, lengths):
        embedding = self.dropout(embedding)

        packed_words = nn.utils.rnn.pack_padded_sequence(embedding, lengths=lengths, enforce_sorted=False, batch_first=True)
        output, (hidden, cell) = self.lstm(packed_words)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, padding_value=0, batch_first=True)

        return output
