import numpy as np
import torch
from torch import nn
from transformers import BertModel, RobertaModel

from models.modules.attention import MultiHeadAttention, PositionalEncoding
from models.modules.gnn_layer import GATConv, GATv2Conv


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(GATLayer, self).__init__()
        self.layer1 = GATConv(in_feats, out_feats, num_heads, residual=True)

    def forward(self, g, h):
        h = self.layer1(g, h)
        return h


class BERTLayer(nn.Module):
    def __init__(self, model_size, num_labels, args):
        super(BERTLayer, self).__init__()
        hidden_size = args['hidden_size']

        self.emb = BertModel.from_pretrained(
            f'bert-{model_size}-uncased',
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout'],
        )
        self.dropout = nn.Dropout(p=args['dropout'])
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_labels)
        self.position = PositionalEncoding(hidden_size)
        self.slf_attn = MultiHeadAttention(8, hidden_size, 64, 64, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        for n, p in self.named_parameters():
            if p.dim() > 1 and 'emb' not in n:
                nn.init.xavier_normal_(p, gain=gain)

    def forward(self, inputs, lens, mask, labels, features=None):
        embs = self.emb(inputs.long(), attention_mask=mask)[0]  # (batch_size, sequence_length, hidden_size)
        if features is not None:
            print("embs", embs.size())
            print("features", features.size())
            embs = torch.cat([embs, features], dim=1)
            print("embs", embs.size())
            x = self.position(embs.size(1) - 7)
            print("x", x.size())
            embs = self.layer_norm(embs + x)
        embs, _ = self.slf_attn(embs, embs, embs)

        h_n = embs.sum(1)
        h_n = self.dropout(h_n)
        h_n = self.relu(h_n)
        logits = self.linear(h_n)
        return logits


class JOINT(nn.Module):
    def __init__(self, fs, model_size, args, num_labels):
        super().__init__()

        self.gat = GATLayer(in_feats=fs, out_feats=768, num_heads=8)
        self.bert = BERTLayer(model_size, args=args, num_labels=num_labels)

    def forward(self, inputs, lens, mask, labels, g, features, url, device):
        gat_emb = self.gat(g, features)
        ids = [i - 1 for i in url]
        ids = torch.from_numpy(np.array(ids)).to(device)
        gat_emb = torch.index_select(gat_emb, 0, ids)
        h = self.bert(inputs, lens, mask, labels, gat_emb)
        return h


class GATv2Layer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(GATv2Layer, self).__init__()
        self.layer1 = GATv2Conv(in_feats, out_feats, num_heads, residual=True)

    def forward(self, g, h):
        h = self.layer1(g, h)
        return h


class JOINTv2(nn.Module):
    def __init__(self, fs, model_size, args, num_labels):
        super().__init__()

        self.gat = GATv2Layer(in_feats=fs, out_feats=768, num_heads=8)
        self.bert = BERTLayer(model_size, args=args, num_labels=num_labels)

    def forward(self, inputs, lens, mask, labels, g, features, url, device):
        gat_emb = self.gat(g, features)
        ids = [i - 1 for i in url]
        ids = torch.from_numpy(np.array(ids)).to(device)
        gat_emb = torch.index_select(gat_emb, 0, ids)
        h = self.bert(inputs, lens, mask, labels, gat_emb)
        return h

class JOINTv2_ROBERTA(nn.Module):
    def __init__(self, fs, model_size, args, num_labels):
        super().__init__()

        self.gat = GATv2Layer(in_feats=fs, out_feats=768, num_heads=8)
        self.bert = ROBERTALayer(model_size, args=args, num_labels=num_labels)

    def forward(self, inputs, lens, mask, labels, g, features, url, device):
        gat_emb = self.gat(g, features)
        ids = [i - 1 for i in url]
        ids = torch.from_numpy(np.array(ids)).to(device)
        gat_emb = torch.index_select(gat_emb, 0, ids)
        h = self.bert(inputs, lens, mask, labels, gat_emb)
        return h

class JOINT_ROBERTA(nn.Module):
    def __init__(self, fs, model_size, args, num_labels):
        super().__init__()

        self.gat = GATLayer(in_feats=fs, out_feats=768, num_heads=8)
        self.bert = ROBERTALayer(model_size, args=args, num_labels=num_labels)

    def forward(self, inputs, lens, mask, labels, g, features, url, device):
        gat_emb = self.gat(g, features)
        ids = [i - 1 for i in url]
        ids = torch.from_numpy(np.array(ids)).to(device)
        gat_emb = torch.index_select(gat_emb, 0, ids)
        h = self.bert(inputs, lens, mask, labels, gat_emb)
        return h

class GAT(nn.Module):
    def __init__(self, fs, model_size, args, num_labels):
        super().__init__()
        hidden_size = args['hidden_size']
        self.gat = GATLayer(in_feats=fs, out_feats=hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.slf_attn = MultiHeadAttention(8, hidden_size, 64, 64, dropout=0.1)
        self.linear = nn.Linear(in_features=hidden_size, out_features=2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        for n, p in self.named_parameters():
            if p.dim() > 1 and 'gat' not in n:
                nn.init.xavier_normal_(p, gain=gain)

    def forward(self, inputs, lens, mask, labels, g, features, url, device):
        gat_emb = self.gat(g, features)
        ids = [i - 1 for i in url]
        ids = torch.from_numpy(np.array(ids)).to(device)
        gat_emb = torch.index_select(gat_emb, 0, ids)
        gat_emb = self.layer_norm(gat_emb)
        gat_emb, _ = self.slf_attn(gat_emb, gat_emb, gat_emb)
        gat_emb = gat_emb.sum(1)
        logits = self.linear(gat_emb)
        return logits


class BERT(nn.Module):
    def __init__(self, fs, model_size, args, num_labels):
        super().__init__()
        self.blstm = BERTLayer(model_size, args=args, num_labels=num_labels)

    def forward(self, inputs, lens, mask, labels, g, features, url, device):
        h = self.blstm(inputs, lens, mask, labels)
        return h

class ROBERTA(nn.Module):
    def __init__(self, fs, model_size, args, num_labels):
        super().__init__()
        self.blstm = ROBERTALayer(model_size, args=args, num_labels=num_labels)

    def forward(self, inputs, lens, mask, labels, g, features, url, device):
        h = self.blstm(inputs, lens, mask, labels)
        return h


class ROBERTALayer(nn.Module):
    def __init__(self, model_size, num_labels, args):
        super(ROBERTALayer, self).__init__()
        hidden_size = args['hidden_size']

        self.emb = RobertaModel.from_pretrained(
            f'roberta-{model_size}',
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout'],
        )
        self.dropout = nn.Dropout(p=args['dropout'])
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_labels)
        self.position = PositionalEncoding(hidden_size)
        self.slf_attn = MultiHeadAttention(8, hidden_size, 64, 64, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        for n, p in self.named_parameters():
            if p.dim() > 1 and 'emb' not in n:
                nn.init.xavier_normal_(p, gain=gain)

    def forward(self, inputs, lens, mask, labels, features=None):
        embs = self.emb(inputs.long(), attention_mask=mask)[0]  # (batch_size, sequence_length, hidden_size)
        if features is not None:
            print("embs", embs.size())
            print("features", features.size())
            embs = torch.cat([embs, features], dim=1)
            print("embs", embs.size())
            x = self.position(embs.size(1) - 7)
            print("x", x.size())
            y = self.position(embs.size(1))
            print("y", y.size())
            embs = self.layer_norm(embs + y)
        embs, _ = self.slf_attn(embs, embs, embs)

        h_n = embs.sum(1)
        h_n = self.dropout(h_n)
        h_n = self.relu(h_n)
        logits = self.linear(h_n)
        return logits