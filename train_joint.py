import json
import os
import random
import sys
import time

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer

from cli import get_args
from data import mydata
from datasets import OLDDataset, ImbalancedDatasetSampler
from graph_data import load_graph
from models.joint import JOINT, JOINTv2, GAT, BERT, ROBERTA, JOINTv2_ROBERTA, JOINT_ROBERTA, TwitterROBERTA, JOINT_TWIT_ROBERTA, JOINTv2_TWIT_ROBERTA
from models.modules.focal_loss import FocalLoss
from trainer_joint import Trainer
from utils import load

# python train_joint.py -bs=64 -lr_other=5e-5 -lr_gat=1e-2 -ep=20  -dr=0.5 -ad=0.1 -hs=768 --model=joint --clip --cuda=0

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    num_trials = args['num_trials']
    args['add_final'] = None
    model_name = args['model']
    model_size = args['model_size']
    truncate = args['truncate']
    epochs = args['epochs']
    lr_other = args['learning_rate_other']
    lr_gat = args['learning_rate_gat']
    wd = args['weight_decay']
    bs = args['batch_size']
    cu = args['cuda']
    ts = args['test_ratio']
    fm = args['feat_model']
    fi = args['feat_init']
    # Fix seed for reproducibility
    seed = args['seed']
    tweet_path = args['tweet_path']
    user_path = args['user_path']
    relationship_path = args['relationship_path']
    log_path = args['log_path']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    # os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    # device = torch.device('cuda:' + cu if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    results_path = os.path.join(log_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    num_labels = 2

    if model_name == 'joint':
        g, _, _, _ = load_graph(tweet_path, user_path, relationship_path, test_size=ts, feat_model=fm, feat_init=fi)
        g = g.to(device)
        features = g.ndata['features']
        features_size = features.size()[1]
        model = JOINT(fs=features_size, model_size=model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'joint_roberta':
        g, _, _, _ = load_graph(tweet_path, user_path, relationship_path, test_size=ts, feat_model=fm, feat_init=fi)
        g = g.to(device)
        features = g.ndata['features']
        features_size = features.size()[1]
        model = JOINT_ROBERTA(fs=features_size, model_size=model_size, args=args, num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif model_name == 'jointv2':
        g, _, _, _ = load_graph(tweet_path, user_path, relationship_path, test_size=ts, feat_model=fm, feat_init=fi)
        g = g.to(device)
        features = g.ndata['features']
        features_size = features.size()[1]
        model = JOINTv2(fs=features_size, model_size=model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'jointv2_roberta':
        g, _, _, _ = load_graph(tweet_path, user_path, relationship_path, test_size=ts, feat_model=fm, feat_init=fi)
        g = g.to(device)
        features = g.ndata['features']
        features_size = features.size()[1]
        model = JOINTv2_ROBERTA(fs=features_size, model_size=model_size, args=args, num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif model_name == 'gat':
        g, _, _, _ = load_graph(tweet_path, user_path, relationship_path, test_size=ts, feat_model=fm, feat_init=fi)
        g = g.to(device)
        features = g.ndata['features']
        features_size = features.size()[1]
        model = GAT(fs=features_size, model_size=model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'bert':
        g = None
        features = None
        model = BERT(fs=None, model_size=model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta':
        g = None
        features = None
        model = ROBERTA(fs=None, model_size=model_size, args=args, num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif model_name == 'twitter_roberta':
        g = None
        features = None
        MODEL = f"cardiffnlp/twitter-roberta-base-offensive"
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
        model = TwitterROBERTA(fs=None, model_size=model_size, args=args, num_labels=num_labels)
    elif model_name == 'joint_twitter_roberta':
        g, _, _, _ = load_graph(tweet_path, user_path, relationship_path, test_size=ts, feat_model=fm, feat_init=fi)
        g = g.to(device)
        features = g.ndata['features']
        features_size = features.size()[1]
        MODEL = f"cardiffnlp/twitter-roberta-base-offensive"
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
        model = JOINT_TWIT_ROBERTA(fs=features_size, model_size=model_size, args=args, num_labels=num_labels)
    elif model_name == 'jointv2_twitter_roberta':
        g, _, _, _ = load_graph(tweet_path, user_path, relationship_path, test_size=ts, feat_model=fm, feat_init=fi)
        g = g.to(device)
        features = g.ndata['features']
        features_size = features.size()[1]
        MODEL = f"cardiffnlp/twitter-roberta-base-offensive"
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
        model = JOINTv2_TWIT_ROBERTA(fs=features_size, model_size=model_size, args=args, num_labels=num_labels)

    # Move model to correct device
    model = model.to(device=device)

    if args['ckpt'] != '':
        model.load_state_dict(load(args['ckpt']))

    _Dataset = OLDDataset

    purl_train, token_ids_train, lens_train, mask_train, labels_train = mydata(path=tweet_path,
                                                                               tokenizer=tokenizer,
                                                                               truncate=truncate,
                                                                               data_type='train', test_size=ts)
    purl_test, token_ids_test, lens_test, mask_test, labels_test = mydata(path=tweet_path,
                                                                          tokenizer=tokenizer,
                                                                          truncate=truncate,
                                                                          data_type='test', test_size=ts)

    datasets = {
        'train': _Dataset(
            input_ids=token_ids_train,
            lens=lens_train,
            mask=mask_train,
            url=purl_train,
            labels=labels_train,
        ),
        'test': _Dataset(
            input_ids=token_ids_test,
            lens=lens_test,
            mask=mask_test,
            url=purl_test,
            labels=labels_test,
        )
    }

    sampler = ImbalancedDatasetSampler(datasets['train'])
    dataloaders = {
        'train': DataLoader(
            dataset=datasets['train'],
            batch_size=bs,
            sampler=sampler
        ),
        'test': DataLoader(dataset=datasets['test'], batch_size=bs)
    }

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss()

    if not (model_name == 'bert' or model_name == 'roberta'):
        layer = list(map(id, model.gat.parameters()))
        base_params = filter(lambda p: id(p) not in layer, model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params},
                                      {'params': model.gat.parameters(), 'lr': lr_gat},
                                      ], lr=lr_other, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_other, weight_decay=wd)
    scheduler = None

    seed_lst = [args['seed']]
    if num_trials > 1:
        random.seed(time.time())
        seed_lst += random.sample(range(sys.maxsize), num_trials)

    combined_metrics = {}
    for s in seed_lst:
        trainer = Trainer(
            model=model,
            epochs=epochs,
            dataloaders=dataloaders,
            features=features,
            criterion=criterion,
            clip=args['clip'],
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_name=model_name,
            final=args['add_final'],
            seed=s,
            g=g,
            patience=args['patience'],
            log_path=log_path
        )

        metrics = trainer.train()
        combined_metrics[f'{seed}'] = metrics

    print('Saving results...')
    data_dump = json.dumps(combined_metrics)

    f = open(os.path.join(results_path, f"{model_name}_x{num_trials}_metrics.json"), "w")
    f.write(data_dump)
    f.close()
