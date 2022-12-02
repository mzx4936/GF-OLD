import pandas as pd
import dgl
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp


def standardize(seq):
    centerized = seq - seq.mean(axis=0)
    normalized = centerized / centerized.std(axis=0)
    return normalized


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def build_graph(nodes, relations, features):
    g = dgl.DGLGraph()
    source = [i-1 for i in relations['user'].tolist()]
    target = [i-1 for i in relations['follow_user'].tolist()]
    # add nodes
    nodes_len = len(nodes)
    g.add_nodes(nodes_len)
    # add relations
    g.add_edges(source, target)
    g = dgl.add_self_loop(g)
    # add features
    g.ndata['features'] = features
    return g, source, target


def load_graph(tweet_path, user_path, relationship_path, test_size=0.3, feat_model='soft', feat_init='non_off'):
    data = pd.read_csv(tweet_path)
    purl = data['user_id'].values
    labels = np.array(data['is_off'].values)
    purl, turl, labels, _ = train_test_split(purl, labels, test_size=test_size, random_state=0)
    nodes = pd.read_csv(user_path)

    relations = pd.read_csv(relationship_path)
    relations.loc[relations.follow_type == 2, 'user'], relations.loc[relations.follow_type == 2, 'follow_user'] = \
        relations.loc[relations.follow_type == 2, 'follow_user'], relations.loc[relations.follow_type == 2, 'user']

    features = []
    ul = pd.DataFrame({'user': purl, 'label': labels})
    labels_count = ul.groupby('user')['label'].value_counts().unstack().fillna(value=0)

    if feat_model == 'soft':
        if feat_init == 'all_zero':
            not_off = off = 0
        elif feat_init == 'all_one':
            not_off = off = 1
        elif feat_init == 'avg':
            not_off = sum(labels_count[0]) / len(labels_count[0])
            off = sum(labels_count[1]) / len(labels_count[1])
        elif feat_init == 'non_off':
            not_off = 1
            off = 1e-6

        labels_count = pd.DataFrame({'user': labels_count.index,
                                     'label': [np.array(i) for i in zip(labels_count[0], labels_count[1])]})
        nodes = pd.merge(nodes, labels_count, on='user', how='left')
        nodes.loc[nodes['label'].isnull(), ['label']] = nodes.loc[nodes['label'].isnull(), 'label'].apply(
            lambda x: np.array([not_off, off]))
        labels = nodes['label'].tolist()
        features = torch.tensor(list(map(list, labels))).float()

    elif feat_model == 'hard':
        labels_count = pd.DataFrame({'user': labels_count.index,
                                     'label': [np.array([0]) if i == 0 else np.array([1]) for i in labels_count[1]]})
        nodes = pd.merge(nodes, labels_count, on='user', how='left')
        nodes.loc[nodes['label'].isnull(), ['label']] = nodes.loc[nodes['label'].isnull(), 'label'].apply(
            lambda x: np.array([1]))
        labels = nodes['label'].tolist()
        features = torch.tensor(labels).float()

    elif feat_model == 'bow':
        from data import mydata
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(f'bert-base-uncased')
        ngrams = mydata(tweet_path, tokenizer, data_type='ngrams')
        user = []
        for url in nodes['user'].tolist():
            feat = ngrams[url]
            user.append([url, feat])
        user.sort(key=lambda x: x[0])
        ngrams = [i[1] for i in user]
        ngrams = sp.csr_matrix(ngrams)
        ngrams = normalize(ngrams)
        features = torch.tensor(np.array(ngrams.todense())).float()

    g, source, target = build_graph(nodes, relations, features)
    print(g)
    return g, source, target, nodes



