import sys

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split


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
    source = [i - 1 for i in relations['user'].tolist()]
    target = [i - 1 for i in relations['follow_user'].tolist()]
    nodes_len = len(nodes)
    # We will assume that the number of nodes is small enough for int32, but just in case, we will assert here.
    assert(nodes_len < 2**31-1)
    g = dgl.graph((source, target), idtype=torch.int32)
    # Add self-loop for each node to preserve old node representation
    #   Note - Using g = g.add_self_loop(g) here raises the following:
    #       DGLError('Invalid key "{}". Must be one of the edge types.'.format(orig_key))
    g.add_edges(g.nodes(), g.nodes())
    g.ndata['features'] = features
    return g, source, target


def load_graph(tweet_path, user_path, relationship_path, test_size=0.3, feat_model='soft', feat_init='non_off'):
    data = pd.read_csv(tweet_path)
    purl = data['user_id'].values
    labels = np.array(data['is_off'].values)
    purl, turl, labels, _ = train_test_split(purl, labels, test_size=test_size, random_state=0)
    nodes = pd.read_csv(user_path)
    nodes = nodes.rename(columns={'user_id': 'user'})

    relations = pd.read_csv(relationship_path)
    relations.loc[relations.follow_type == 2, 'user'], relations.loc[relations.follow_type == 2, 'follow_user'] = \
        relations.loc[relations.follow_type == 2, 'follow_user'], relations.loc[relations.follow_type == 2, 'user']

    features = []
    ul = pd.DataFrame({'user': purl, 'label': labels})
    labels_count = ul.groupby('user')['label'].value_counts().unstack().fillna(value=0)

    if feat_model == 'soft':
        not_off = None
        off = None
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
        labels_count = pd.DataFrame(data={'user': labels_count.index,
                                          'not_off': labels_count[0],
                                          'off': labels_count[1]},
                                    index=[i for i in range(len(labels_count.index))])
        nodes = pd.merge(nodes, labels_count, on='user', how='left')
        nodes.loc[nodes['not_off'].isnull(), ['not_off']] = not_off
        nodes.loc[nodes['off'].isnull(), ['off']] = off
        labels = nodes[['not_off', 'off']].to_numpy()
        features = torch.tensor(list(map(list, labels))).float()

    elif feat_model == 'hard':
        labels_count = pd.DataFrame({'user': labels_count.index,
                                     'not_off': labels_count[0],
                                     'off': labels_count[1]})
        nodes = pd.merge(nodes, labels_count, on='user', how='left')
        nodes.loc[nodes['not_off'].isnull(), ['not_off']] = 1
        nodes.loc[nodes['off'].isnull(), ['off']] = 1
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
    assert(features.size()[0] == len(nodes))
    g, source, target = build_graph(nodes, relations, features)
    print(g)
    return g, source, target, nodes
