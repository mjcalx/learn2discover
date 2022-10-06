import os
import sys
import traceback
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime
import re
import os
from random import shuffle
from collections import defaultdict

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
current_path = os.path.join(root_path, 'learn2discover')
lib_path = os.path.join(root_path,'lib')

for path in [root_path, current_path, lib_path]:
    if path not in sys.path:
        sys.path.append(path)
sys.path.index(current_path)
sys.path.index(lib_path)

from configs.config_manager import ConfigManager
from utils.logging_utils import LoggingUtils, Verbosity
from loggers.logger_factory import LoggerFactory
from data.schema import VarType
from data.dataset_manager import DatasetManager
from data.data_classes import ParamType, Label
from oracle.query_strategies.query_strategy_factory import QueryStrategyFactory
from oracle.stopping_criteria.stopping_criterion_factory import StoppingCriterionFactory
from oracle.l2d_classifier import L2DClassifier


def main():
    learn_to_discover = Learn2Discover()
    learn_to_discover.run()

class Learn2Discover:
    """
    Learn2Discover
    """
    def __init__(self, workspace_dir=os.getcwd()):
        self.config_manager = ConfigManager(workspace_dir)
        LoggingUtils.get_instance().debug('Loaded Configurations.')
        try:
            self.logger = LoggerFactory.get_logger(__class__.__name__)
            self.logger.debug('Loaded Logger.')
            
            self.dataset_manager = DatasetManager()
            self.dataset = self.dataset_manager.data
            self.logger.debug('Loaded DatasetManager.')
            
            self.query_strategy = QueryStrategyFactory().get_strategy(self.config_manager.query_strategy)
            self.stopping_criterion = StoppingCriterionFactory().get_stopping_criterion(self.config_manager.stopping_criterion)

            _uf = self.config_manager.unlabelled_fraction
            self.unlabelled_fraction = _uf if _uf is not None else 0
            self.test_fraction = self.config_manager.test_fraction

            # Human-specific params
            if self.config_manager.has_human_in_the_loop: 
                self._get_annotations = self._get_annotations_human
            else:
                self._get_annotations = self._get_annotations_auto

        except BaseException as e:
            self.logger.error(f'{e}')
            self.logger.error('Exiting...')
            print(traceback.format_exc())
            exit()

    def run(self):
        

        while not self.stopping_criterion():
            self.stopping_criterion.report()
            self.active_learning_loop()
        
        self.logger.info('Stopping criterion reached. Beginning evaluation...', verbosity=Verbosity.BASE)
        test_idxs_shuffled = self.dataset_manager.shuffle(self.dataset.evaluation_data.index)
        fscore, auc = self.classifier.evaluate_model(test_idxs_shuffled)
        model_path = self.classifier.save_model(fscore, auc)
        self.logger.info(f"[fscore, auc] = [{fscore}, {auc}]")
        self.logger.info(f"Model saved to:  {model_path}")

    def active_learning_loop(self):
        """
        Perform the logic of the active learning loop
        """
        ##### TRAIN MODEL USING SET OF CURRENTLY LABELLED DATA #####
        numerical_data = self.dataset.get_tensors_of_type(VarType.NUMERICAL)
        self.classifier = L2DClassifier(numerical_data.shape[1])
        
        train_idxs_shuffled = self.dataset_manager.shuffle(self.dataset.training_data.index)
        test_idxs_shuffled = self.dataset_manager.shuffle(self.dataset.evaluation_data.index)
        self.classifier.fit(train_idxs_shuffled)

        
        ##### TAKE SAMPLE OF UNLABELLED DATA AND PREDICT THE LABELS #####
        self.classifier.eval()  # stop training in order to query single samples

        sampled_idxs = self.dataset_manager.choose_random_unlabelled(self.dataset.unlabelled_idxs)
        random_items = self.dataset.unlabelled_data.loc[sampled_idxs]

        _m = 'run(): selected random sample of {} unlabelled instances'
        self.logger.debug(_m.format(len(sampled_idxs)), verbosity=Verbosity.CHATTY)
        self.logger.debug(f'First 5 sampled: \n{random_items[:5]}', verbosity=Verbosity.CHATTY)


        ##### APPLY QUERY STRATEGY #####
        # Sample unlabelled items per iteration based on query strategy
        sample_unlabelled = self.query_strategy.query(self.classifier, random_items)

        idxs_shuffled = self.dataset_manager.shuffle(sample_unlabelled.index)
        shuffled_sample_unlabelled = sample_unlabelled.loc[idxs_shuffled]


        ##### QUERY ORACLE FOR TRUE LABELS FOR THE INSTANCES #####
        # pass responsibility for labelling to attached oracle
        annotated_data = self._get_annotations(shuffled_sample_unlabelled)


        ##### ADD LABELLED INSTANCES TO THE SET OF LABELLED DATA #####
        self._update_training_data(annotated_data)
        self.classifier.train()  # stop querying the model and continue training

        """Train model on the given training_data
        Tune with the validation_data
        Evaluate accuracy with the evaluation_data
        """



    def _update_training_data(self, annotated_data: pd.DataFrame) -> None:
        FAIRNESS = ParamType.FAIRNESS.value
        FAIR     = Label.FAIR.value
        UNFAIR   = Label.UNFAIR.value

        _select = lambda label : annotated_data[FAIRNESS][FAIRNESS][lambda x : x == label]
        annotated_data_fair   = _select(FAIR)
        annotated_data_unfair = _select(UNFAIR)

        _old_len_fair   = len(self.dataset.training_data_fair)
        _old_len_unfair = len(self.dataset.training_data_unfair)
        _m =  'will add annotations:\n'
        _m += '\t{} fair instances   + {} annotated "fair" instances   = {}  updated fair instance count\n'
        _m += '\t{} unfair instances + {} annotated "unfair" instances = {}  updated unfair instance count\n'
        self.logger.debug(_m.format(
            _old_len_fair,   len(annotated_data_fair),   _old_len_fair + len(annotated_data_fair),
            _old_len_unfair, len(annotated_data_unfair), _old_len_fair + len(annotated_data_fair),
            verbosity = Verbosity.CHATTY
        ))
        self.dataset.set_training_data(self.dataset.training_data.index.union(annotated_data.index))

    def _get_annotations(self, unlabelled_data: pd.DataFrame) -> pd.DataFrame:
        if self.config_manager.has_human_in_the_loop:
            annotated = self._get_annotations_human(unlabelled_data)
        else:
            annotated = self._get_annotations_auto(unlabelled_data)
        assert len(set(annotated_data.index).intersection(set(self.dataset.training_data.index))) == 0
        return annotated

    def _get_annotations_auto(self, unlabelled_data: pd.DataFrame) -> pd.DataFrame:
        # Nothing to do
        return unlabelled_data

    def _get_annotations_human(self) -> pd.DataFrame:
        # Get annotation data from annotator via command line
        raise NotImplementedError

if __name__=="__main__":
    main()