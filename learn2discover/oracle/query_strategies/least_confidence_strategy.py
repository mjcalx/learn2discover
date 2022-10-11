import pandas as pd
import torch
import torch.nn as nn
from oracle.query_strategies.query_strategy import QueryStrategy
from data.enum import VarType
from utils.logging_utils import Verbosity
from utils.classifier_utils import ClassifierUtils

class LeastConfidenceStrategy(QueryStrategy):
    def __init__(self):
        super(LeastConfidenceStrategy, self).__init__()

    @property
    def name(self):
        return 'Least Confidence Strategy'

    def query(self, classifier: nn.Module, unlabelled_data: pd.DataFrame, number: int=80, limit: int=10000) -> pd.DataFrame:
        """
        Assumption: the final layer of the classifier is a LogSoftmax layer
        """
        num_instances = len(unlabelled_data)
        _m = 'query(): received {} unlabelled instances.'
        self.logger.debug(_m.format(num_instances))
        confidences = []
        if limit == -1 or limit >= num_instances: 
            _m =  "query(): evaluating confidence for all {} unlabelled "
            _m += "instances (this might take a while)..."
            self.logger.debug(_m.format(num_instances))
            idxs = unlabelled_data.index
        else: 
            # only apply the model to a limited number of items
            idxs = self.dataset_manager.shuffle(unlabelled_data)[:limit]

        tensors = self.dataset_manager.tensor_data.loc(idxs)
        categorical_tensors = tensors[VarType.CATEGORICAL]
        numerical_tensors = tensors[VarType.NUMERICAL]
        
        # Get the log probabilities from the model and map to a confidence in range [0.5, 1]
        with torch.no_grad():
            for i in range(num_instances):
                id = idxs[i]
                self.logger.debug(f'ID={id}', verbosity=Verbosity.TALKATIVE)

                y_pred = classifier(categorical_tensors[None, i], numerical_tensors[None, i])
                self.logger.debug(f'LOG_PROBS: {y_pred}', verbosity=Verbosity.CHATTY)
                conf = ClassifierUtils.get_confidence_from_log_probs(y_pred)
                confidences.append((id,conf))

        # Return the ids of least confidence from those sampled
        confidences.sort(key=lambda x: x[1])
        return_idxs = list(zip(*confidences))[0][:number:]
        _m = 'query(): top results: {}'
        self.logger.debug(_m.format(confidences[:5:]))
        return unlabelled_data.loc[pd.Index(return_idxs)]