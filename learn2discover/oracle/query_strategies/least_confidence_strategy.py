import pandas as pd
import torch
import torch.nn as nn
from oracle.query_strategies.query_strategy import QueryStrategy
from data.data_classes import VarType
from utils.logging_utils import Verbosity
# from data.dataset_manager import DatasetManager

class LeastConfidenceStrategy(QueryStrategy):
    def __init__(self):
        super(LeastConfidenceStrategy, self).__init__()
        self.already_labelled = pd.Index([])

    def __name__(self):
        return 'Least Confidence Strategy'

    def query(self, classifier: nn.Module, unlabelled_data: pd.DataFrame, number: int=80, limit: int=10000) -> pd.DataFrame:
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
            shuffled = self.dataset_manager.shuffle(unlabeled_data)
            idxs = shuffled[:limit].index

        tensors = self.dataset_manager.tensor_data.loc(idxs)
        categorical_tensors = tensors[VarType.CATEGORICAL]
        numerical_tensors = tensors[VarType.NUMERICAL]
        
        # for each sampled item:
        #     ignore if already labelled
        #     get the score from the model
        with torch.no_grad():
            for i in range(num_instances):
                id = idxs[i]
                self.logger.debug(f'ID={id}', verbosity=Verbosity.TALKATIVE)

                if id in self.already_labelled:
                    continue
                
                ###############################

                output = classifier(categorical_tensors[None, i], numerical_tensors[None, i])
                probs = torch.nn.functional.softmax(output, dim=1)
                prob_fair = probs[0][0] # TODO confirm not [0][1]
                self.logger.debug(f'PROBS: {probs}', verbosity=Verbosity.TALKATIVE)
                # feature_vector = self.make_feature_vector(text.split(), self.feature_index)
                # log_probs = self(feature_vector)

                # get confidence that it is related
                # prob_related = math.exp(log_probs.data.tolist()[0][1]) 
                
                if prob_fair < 0.5:
                    confidence = 1 - prob_fair
                else:
                    confidence = prob_fair

                confidences.append((id,confidence))

        # todo return lowest (highest?) confidence values, up to `number`
        confidences.sort(key=lambda x: x[1])
        return_idxs = list(zip(*confidences))[0][:number:]
        _m = 'query(): top results: {}'
        self.logger.debug(_m.format(confidences[:5:]))
        return unlabelled_data.loc[pd.Index(return_idxs)]