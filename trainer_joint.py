# Built-in libraries
import copy
import datetime
from typing import Dict, List
# Third-party libraries
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
from tqdm import tqdm
# Local files
from utils import save
import torch.nn.functional as F
import time

class Trainer():
    '''
    The trainer for training models.
    It can be used for both single and multi task training.
    Every class function ends with _m is for multi-task training.
    '''

    def __init__(
            self,
            model,
            epochs,
            dataloaders,
            features,
            criterion,
            clip,
            optimizer,
            scheduler,
            device,
            model_name,
            final,
            seed,
            g,
            patience,
            log_path
    ):
        self.model = model
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.features = features
        self.criterion = criterion
        self.clip = clip
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.final = final
        self.seed = seed
        self.g = g
        self.patience = patience
        self.datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')
        self.log_path = log_path

        # Evaluation results
        self.train_losses = []
        self.test_losses = []
        self.train_f1 = []
        self.test_f1 = []
        self.best_train_f1 = 0.0
        self.best_test_f1 = 0.0

        self.train_accuracy = []
        self.test_accuracy = []
        self.best_train_accuracy = 0.0
        self.best_test_accuracy = 0.0

        self.train_recall = []
        self.test_recall = []
        self.best_train_recall = 0.0
        self.best_test_recall = 0.0

        self.train_precision = []
        self.test_precision = []
        self.best_train_precision = 0.0
        self.best_test_precision = 0.0

        self.best_AUC = 0.0

        self.epoch = 0
        self.best_epoch = []
        self.time = []




    def train(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch()
            self.test()
            print(f'Best epoch: {self.best_epoch}')
            print(f'f1: {self.test_f1}')
            print(f'accuracy: {self.test_accuracy}')
            # print(f'fpr: {self.fpr}')
            # print(f'tpr: {self.tpr}')
            print(f'Best test f1: {self.best_test_f1:.4f}')
            print(f'Best test accuracy: {self.best_test_accuracy:.4f}')
            print(f'Best test recall: {self.best_test_recall:.4f}')
            print(f'Best test precision: {self.best_test_precision:.4f}')
            print(f'Best AUC: {self.best_AUC:.4f}')
            print('=' * 20)
            if epoch - self.best_epoch[-1][0] == self.patience - 1:
                print('============early stopping==============')
                print(f'now: {epoch}    last: {self.best_epoch[-1][0]}')
                break

        metrics = {
            'train_loss': self.train_losses,
            'test_loss': self.test_losses,
            'train_acc': self.train_accuracy,
            'test_acc': self.test_accuracy,
            'train_recall': self.train_recall,
            'test_recall': self.test_recall,
            'train_precision': self.train_precision,
            'test_precision': self.test_precision,
            'train_f1': self.train_f1,
            'test_f1': self.test_f1,
            'best_train_f1': self.best_train_f1,
            'best_test_f1': self.best_test_f1,
        }
        return metrics

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        start = time.time()
        for inputs, lens, mask, labels, url in tqdm(dataloader, desc='Training'):
            iters_per_epoch += 1
            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                logits = self.model(inputs, lens, mask, labels, self.g, self.features, url, self.device)
                _loss = self.criterion(logits, labels)
                loss += _loss.item()
                y_pred = logits.argmax(dim=1).cpu().numpy()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        end = time.time()
        self.time.append(end-start)
        print(f'process time: {sum(self.time)/(self.epoch+1)}')

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')
        accuracy = accuracy_score(labels_all, y_pred_all)
        recall = recall_score(labels_all, y_pred_all, average='macro')
        precision = precision_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-accuracy = {accuracy:.4f}')
        print(f'Macro-recall = {recall:.4f}')
        print(f'Macro- precision = { precision:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.train_losses.append(loss)
        self.train_accuracy.append(float('%.4f' % accuracy))
        self.train_recall.append(float('%.4f' % recall))
        self.train_precision.append(float('%.4f' % precision))
        self.train_f1.append(float('%.4f' % f1))
        if f1 > self.best_train_f1:
            self.best_train_f1 = f1

    def test(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels, url in tqdm(dataloader, desc='Testing'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(inputs, lens, mask, labels, self.g, self.features, url, self.device)
                _loss = self.criterion(logits, labels)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                y_prob = torch.index_select(F.softmax(logits, dim=1).cpu(), 1, torch.tensor([1])).numpy()
                loss += _loss.item()

                if y_pred_all is None:
                    y_pred_all = y_pred
                    y_prob_all = y_prob
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))
                    y_prob_all = np.concatenate((y_prob_all, y_prob))

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')
        accuracy = accuracy_score(labels_all, y_pred_all)
        recall = recall_score(labels_all, y_pred_all, average='macro')
        precision = precision_score(labels_all, y_pred_all, average='macro')

        fpr, tpr, thresholds = roc_curve(labels_all, y_prob_all)

        AUC = auc(fpr, tpr)
        fpr = [float('%.4f'%i) for i in fpr]
        tpr = [float('%.4f'%i) for i in tpr]


        print(f'loss = {loss:.4f}')
        print(f'Macro-accuracy = {accuracy:.4f}')
        print(f'Macro-recall = {recall:.4f}')
        print(f'Macro- precision = {precision:.4f}')
        print(f'Macro-F1 = {f1:.4f}')
        print(f'AUC = {AUC:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append(float('%.4f' % f1))
        self.test_accuracy.append(float('%.4f' % accuracy))

        if f1 > self.best_test_f1:
            self.best_test_f1 = f1
            self.best_test_accuracy = accuracy
            self.best_test_recall = recall
            self.best_test_precision = precision
            self.best_epoch.append((self.epoch, '%.4f' % f1))
            self.save_model()
            self.best_AUC = AUC
            self.fpr = fpr
            self.tpr = tpr

    def save_model(self):
        print('Saving model...')

        filename = f'{self.log_path}/saved_models/{self.model_name}_{self.seed}_best.pt'
        save(copy.deepcopy(self.model.state_dict()), filename)