import os
import re
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_curve, 
    roc_auc_score, precision_score, recall_score, f1_score
)
from typing import List
from pathlib import Path

from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager
from data.data_classes import ParamType, Label, VarType
from loggers.logger_factory import LoggerFactory


class L2DClassifier(nn.Module):
    def __init__(self, embedding_size, num_numerical_cols):
        super(L2DClassifier, self).__init__()
        cfg = ConfigManager.get_instance()
        self.datamgr = DatasetManager.get_instance()
        self.logger = LoggerFactory.get_logger(__class__.__name__)

        self.epochs = cfg.epochs
        self.learning_rate = cfg.learning_rate
        self.selections_per_epoch = cfg.selections_per_epoch
        self.layers = cfg.layers
        self.dropout_rate = cfg.dropout_rate

        self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(self.dropout_rate)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        self.stack = None
        _td = self.datamgr.tensor_data
        num_categorical_cols = sum([nf for ni, nf in embedding_size])
        self._build(len_input=num_categorical_cols+len(_td.numerical_columns))
        self.logger.debug(f'NUM COLUMNS: CAT {len(_td.categorical_columns)}, NUM {len(_td.numerical_columns)}', verbosity=1)
        self.logger.debug(f'INIT EMBEDDING SUM CATEGORICAL: {num_categorical_cols}', verbosity=1)

        #todo what are the parameters?
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.NLLLoss()
        self.metrics = ['fscore', 'auc']
    
    def _build(self, len_input: int):
        all_layers = []
        for i in self.layers:
            all_layers.append(nn.Linear(len_input, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(self.dropout_rate))
            len_input = i
        all_layers.append(nn.Linear(self.layers[-1], len(Label)))

        self.stack = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        # Define how data is passed through the model
        m = 'In forward(): \n\tx_categorical:\n{}\n\tx_numerical:\n{}'
        self.logger.debug(m.format(x_categorical, x_numerical), verbosity=3)
        n = 'In forward(): x_categorical has {}, x_numerical has {}'
        self.logger.debug(n.format(x_categorical.size(), x_numerical.size()), verbosity=3)

        embeddings = []
        for i,e in enumerate(self.embeddings):
            self.logger.debug(f'In forward(): {i},{e}', verbosity=1)
            self.logger.debug(f'\n{e(x_categorical[:,i])}', verbosity=3)
            embeddings.append(e(x_categorical[:,i]))
        # TODO refactor
        x = torch.cat(embeddings,1)
        x = self.embedding_dropout(x)
        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.stack(x)
        return x

    def fit(self, idxs: pd.Index, epochs=None) -> None:
        self.logger.debug("Starting training...", verbosity=1)
        self.len_data = len(idxs)
        epochs = epochs if epochs is not None else self.epochs
        
        _dict_tensors = self.datamgr.tensor_data.loc(idxs)
        i = 100 # TODO index out of range in self
        categorical_data = _dict_tensors[VarType.CATEGORICAL][:i]
        numerical_data     = _dict_tensors[VarType.NUMERICAL][:i]
        labels        = self.datamgr.tensor_data.torch_dataset.labels[idxs][:i]

        m = 'SHAPES: {} (categorical) | {} (numerical) {} (labels)'
        self.logger.debug(m.format(categorical_data.shape, numerical_data.shape, labels.shape))
        aggregated_losses = []
        # TODO 
        # (1) perform selections per epoch
        # (2) shuffle pre-epoch selections
        for e in range(epochs):
            e += 1
            y_pred = self(categorical_data, numerical_data)
            # compute loss function, do backward pass, and update the gradient
            single_loss = self.loss_function(y_pred, labels)
            aggregated_losses.append(single_loss)

            self.optimizer.zero_grad()
            single_loss.backward()
            self.optimizer.step()
            self.logger.debug(f'epoch: {e:3} loss: {single_loss.item():10.10f}', verbosity=1)
        self.logger.debug(f'FINAL EPOCH: {e:3}')
        self.logger.debug(f'TRAINING LOSS: {single_loss.item():10.10f}')

        #TODO EXTRACT
        plt.plot(range(epochs), [i.item() for i in aggregated_losses])
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.savefig('loss-vs-epoch.png')
        plt.clf()
        return

        FAIRNESS = ParamType.FAIRNESS.value
        FAIR = Label.FAIR.value
        UNFAIR = Label.UNFAIR.value

        for epoch in range(epochs):
            print("Epoch: "+str(epoch))
            current = 0
            # make a subset of data to use in this epoch
            # with an equal number of items from each label

            X = self.datamgr.shuffle(training_data) #randomize the order of the training data
            _index_fn = lambda label : X[FAIRNESS][lambda x : x[FAIRNESS] == label].index
            fair = X.loc[_index_fn(FAIR)]
            unfair = X.loc[_index_fn(UNFAIR)]
            
            # Select an equal number of samples from each set for this epoch
            epoch_data = pd.concat([fair[:self.selections_per_epoch], unfair[:self.selections_per_epoch]])
            epoch_data = self.datamgr.shuffle(epoch_data) 

    def evaluate_model(self, test_idxs: pd.Index) -> List[float]:
        """Evaluate the model on the held-out evaluation data
        Return the f-value for disaster-related and the AUC
        """
        i= 100 # TODO batch? Duplications
        _dict_tensors = self.datamgr.tensor_data.loc(test_idxs)
        categorical_data = _dict_tensors[VarType.CATEGORICAL][:i]
        numerical_data     = _dict_tensors[VarType.NUMERICAL][:i]
        labels        = self.datamgr.tensor_data.tensor_dataset.labels[test_idxs][:i]

        with torch.no_grad():
            y_val = self(categorical_data, numerical_data)
            loss  = self.loss_function(y_val, labels)
        y_val = np.argmax(y_val, axis=1)
        auc = roc_auc_score(labels, y_val)
        fpr, tpr, _ = roc_curve(labels, y_val)

        #TODO EXTRACT
        plt.plot(fpr, tpr, label="AUC="+str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('roc.png')

        self.logger.debug(f'TEST LOSS: {loss:.8f}')
        self.logger.debug(f'CONFUSION MATRIX:\n{confusion_matrix(labels, y_val)}')
        self.logger.debug(f'CLASSIFICATION REPORT: \n{classification_report(labels, y_val)}')
        self.logger.debug(f'ACCURACY: {accuracy_score(labels, y_val)}')
        self.logger.debug(f'AUC: {auc}')

        # precision = precision_score(labels, y_val)
        # recall = recall_score(labels, y_val)
        fscore = f1_score(labels, y_val)
        return[fscore, auc]

    def save_model(self, fscore, auc):
        fscore = round(fscore,3)
        auc = round(auc,3)
        cfg = ConfigManager.get_instance()

        # save model to path that is alphanumeric and includes number of items and accuracies in filename
        timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
        training_size = "_"+str(self.len_data)
        accuracies = str(fscore)+"_"+str(auc)

        Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(cfg.model_path,timestamp+accuracies+training_size+".params")

        torch.save(self.state_dict(), model_path)
        return model_path
