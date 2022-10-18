import os
import re
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
)
from typing import List, Tuple, Dict
from pathlib import Path

from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager
from data.enum import ParamType, Label, VarType
from loggers.logger_factory import LoggerFactory
from utils.logging_utils import Verbosity
from utils.observer import Subject
from utils.classifier_utils import ClassifierUtils

class HalvedCeilingFn:
    """
    A sample embedding size function used for this classifier. 

    Calculate embedding as the ceiling of half the number of categories in a 
    column, capped by `upper_bound`.

    Return a list of 2-Tuples of form (old_size, embedding_size), where 
    old_size is the number of categories in a given column.
    """
    def __call__(self, num_categories_per_col: List[int], upper_bound: int=50) -> List[Tuple[int,int]]:
        return [(n, min(upper_bound, (n+1)//2)) for n in num_categories_per_col]

    @property
    def name(self):
        return 'Halved Ceiling Embedding'

class L2DClassifier(nn.Module, Subject):
    def __init__(self, num_numerical_cols):
        super(L2DClassifier, self).__init__()
        cfg = ConfigManager.get_instance()
        self.datamgr = DatasetManager.get_instance()
        self.logger = LoggerFactory.get_logger(__class__.__name__)

        self.epochs = cfg.epochs
        self.learning_rate = cfg.learning_rate
        self.selections_per_epoch = cfg.selections_per_epoch
        self.layers = cfg.layers
        self.dropout_rate = cfg.dropout_rate

        ####### EMBEDDING SIZE FUNCTION #######
        self.embedding_size_function = HalvedCeilingFn()
        embedding_fn_settings = {'num_categories_per_col' : self.datamgr.data.categorical_column_sizes}

        _m = 'Using embedding size function : {}'
        self.logger.debug(_m.format(self.embedding_size_function.name), verbosity=Verbosity.BASE)

        categorical_embedding_size = self.embedding_size_function(**embedding_fn_settings)
        ########################################

        self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in categorical_embedding_size])
        self.embedding_dropout = nn.Dropout(self.dropout_rate)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)
        self.stack = None

        _td = self.datamgr.data
        num_categorical_cols = sum([nf for ni, nf in categorical_embedding_size])
        self._build(len_input=num_categorical_cols+len(_td.numerical_columns))
        self.logger.debug(f'NUM COLUMNS: CAT {len(_td.categorical_columns)}, NUM {len(_td.numerical_columns)}', verbosity=Verbosity.CHATTY)
        self.logger.debug(f'INIT EMBEDDING SUM CATEGORICAL: {num_categorical_cols}', verbosity=Verbosity.CHATTY)

        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.NLLLoss()
        self.metrics = ['fscore', 'auc']

        self._iteration = 0
        self._data = None
    
    @property
    def data(self):
        return self._data
    
    def _build(self, len_input: int):
        all_layers = []
        for i in self.layers:
            all_layers.append(nn.Linear(len_input, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(self.dropout_rate))
            len_input = i
        all_layers.append(nn.Linear(self.layers[-1], len(Label)))
        all_layers.append(nn.LogSoftmax(1))

        self.stack = nn.Sequential(*all_layers)

    def forward(self, x_categorical: torch.Tensor, x_numerical: torch.Tensor):
        # Define how data is passed through the model
        m = 'In forward(): \n\tx_categorical:\n{}\n\tx_numerical:\n{}'
        self.logger.debug(m.format(x_categorical, x_numerical), verbosity=Verbosity.INCESSANT)
        n = 'In forward(): x_categorical has {}, x_numerical has {}'
        self.logger.debug(n.format(x_categorical.size(), x_numerical.size()), verbosity=Verbosity.VERBOSE)

        concatenated = torch.cat([
            self._pass_categorical(x_categorical), 
            self._pass_numerical(x_numerical)
        ], 1)

        return self.stack(concatenated)

    def _pass_categorical(self, x_categorical: torch.Tensor):
        embeddings = []
        for i,e in enumerate(self.embeddings):
            self.logger.debug(f'In forward(): {i},{e}', verbosity=Verbosity.TALKATIVE)
            self.logger.debug(f'\n{e(x_categorical[:,i])}', verbosity=Verbosity.INCESSANT)
            embeddings.append(e(x_categorical[:,i]))
        x = self.embedding_dropout(torch.cat(embeddings,1))
        return x

    def _pass_numerical(self, x_numerical: torch.Tensor):
        x = self.batch_norm_num(x_numerical)
        return x

    def fit(self, idxs: pd.Index, epochs=None) -> None:
        self.logger.debug("Starting training...", verbosity=Verbosity.CHATTY)
        self.logger.debug(f'Will train with learning_rate={self.learning_rate}', verbosity=Verbosity.BASE)
        self._iteration += 1
        self.len_data = len(idxs)
        epochs = epochs if epochs is not None else self.epochs
        get_categorical = lambda _dct_tensors : _dct_tensors[VarType.CATEGORICAL]
        get_numerical  = lambda _dct_tensors : _dct_tensors[VarType.NUMERICAL]

        aggregated_losses = []
        # (1) perform selections per epoch
        # (2) shuffle pre-epoch selections
        for e in range(epochs):
            idxs = self.datamgr.shuffle(idxs)
            _dict_tensors = self.datamgr.data.loc(idxs)
            categorical_data = get_categorical(_dict_tensors)[:self.selections_per_epoch:]
            numerical_data   = get_numerical(_dict_tensors)[:self.selections_per_epoch:]
            labels           = self.datamgr.data.tensor_labels[idxs][:self.selections_per_epoch:]

            m = 'SHAPES: {} (categorical) | {} (numerical) {} (labels)'
            self.logger.debug(m.format(categorical_data.shape, numerical_data.shape, labels.shape), verbosity=Verbosity.INCESSANT)

            e += 1
            y_pred = self(categorical_data, numerical_data)
            # compute loss function, do backward pass, and update the gradient
            single_loss = self.loss_function(y_pred, labels)
            aggregated_losses.append(single_loss)

            self.optimizer.zero_grad()
            single_loss.backward()
            self.optimizer.step()

            self.logger.debug(f'epoch: {e:3} loss: {single_loss.item():10.10f}', verbosity=Verbosity.CHATTY)

        # Evaluate model at each iteration
        test_idxs_shuffled = self.datamgr.shuffle(self.datamgr.data.test_data.index)
        self.evaluate_model(test_idxs_shuffled)


    def evaluate_model(self, eval_idxs: pd.Index, final_report=False) -> Dict[str, object]:
        """Evaluate the model and return evaluation metrics"""
        
        self.eval()

        _dict_tensors = self.datamgr.data.loc(eval_idxs)
        categorical_data = _dict_tensors[VarType.CATEGORICAL]
        numerical_data     = _dict_tensors[VarType.NUMERICAL]
        labels        = self.datamgr.data.tensor_labels[eval_idxs]

        with torch.no_grad():
            log_probs = self(categorical_data, numerical_data)
            loss  = self.loss_function(log_probs, labels)
        y_val = np.argmax(log_probs, axis=1)
        auc = roc_auc_score(labels, y_val)
        fpr, tpr, _ = roc_curve(labels, y_val)

        accuracy = accuracy_score(labels, y_val)
        precision = precision_score(labels, y_val)
        recall = recall_score(labels, y_val)
        fscore = f1_score(labels, y_val)

        if not final_report:
            vloss = loss.item()
            confidences = ClassifierUtils.get_confidence_from_log_probs(log_probs)
            mean_confidence = sum(confidences)/len(confidences)
            accuracy = accuracy_score(labels, y_val)
            self._data = (self._iteration, vloss, self.datamgr.data.training_data.count, accuracy, precision, mean_confidence)
            self.notify()

        results = {'f': fscore, 'auc': auc, 'loss':loss, 
                   'y_pred':y_val, 'y':labels, 
                   'tpr':tpr, 'fpr':fpr,
                   'acc': accuracy, 'precision':precision, 'recall':recall}
        
        self.train()
        return results

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
