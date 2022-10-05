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

            self.test_fraction = self.config_manager.test_fraction
            
            _uf = self.config_manager.unlabelled_fraction
            self.unlabelled_fraction = _uf if _uf is not None else 0
            self.min_evaluation_items = self.config_manager.min_evaluation_items
            self.min_training_items = self.config_manager.min_training_items

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
        single_iteration_test_flag = False
        while not self.stopping_criterion() and not single_iteration_test_flag:
            print('entering')
            if self.dataset.evaluation_count <  self.min_evaluation_items:
                self._fill_evaluation_data()
         
            elif self.dataset.training_count < self.min_training_items:
                self._fill_training_data()

            else:
                self._learn()

            if self.dataset.training_count > self.min_training_items:
                self._annotate_and_retrain()
            
            single_iteration_test_flag = True
        self.logger.info('Stopping criterion reached. Exiting...', verbosity=Verbosity.BASE)

    def _train_and_evaluate(self, training_idxs: pd.Index, test_idxs: pd.Index) -> str:
        train_idxs_shuffled = self.dataset_manager.shuffle(training_idxs)
        test_idxs_shuffled = self.dataset_manager.shuffle(test_idxs)
        
        self.classifier.fit(train_idxs_shuffled)  # Training
        fscore, auc = self.classifier.evaluate_model(test_idxs_shuffled)  # Evaluation
        
        model_path = self.classifier.save_model(fscore, auc)
        return model_path

    def _learn(self):
        # Train new model with current training data
        embedding_size_function = lambda num_categories_per_col: [(n, min(50, (n+1)//2)) for n in num_categories_per_col]
        numerical_data = self.dataset.get_tensors_of_type(VarType.NUMERICAL)

        embedding_sizes = self.dataset.categorical_embedding_sizes
        numerical_data = self.dataset.get_tensors_of_type(VarType.NUMERICAL)

        self.classifier = L2DClassifier(embedding_sizes, numerical_data.shape[1])
        
        model_path = self._train_and_evaluate(
            training_idxs=self.dataset.training_data.index,
            test_idxs=self.dataset.evaluation_data.index
        )

        self.classifier.load_state_dict(torch.load(model_path))
        
        # stop training in order to query single samples
        self.classifier.eval()

        # get 100 items per iteration with the following breakdown of strategies:
        sampled_idxs = self.dataset_manager.choose_random_unlabelled(self.dataset.unlabelled_idxs)
        _m = 'run(): selected random sample of {} unlabelled instances'
        self.logger.debug(_m.format(len(sampled_idxs)))

        random_items = self.dataset.unlabelled_data.loc[sampled_idxs]
        self.logger.debug(f'First 5 sampled: \n{random_items[:5]}', verbosity=Verbosity.CHATTY)
        
        sample_unlabelled = self.query_strategy.query(self.classifier, random_items)

        # stop using existing model for queries and continue training
        self.classifier.train()
 
        FAIRNESS = ParamType.FAIRNESS.value
        FAIR     = Label.FAIR.value
        UNFAIR   = Label.UNFAIR.value

        idxs_shuffled = self.dataset_manager.shuffle(sample_unlabelled.index)
        shuffled_sample_unlabelled = sample_unlabelled.loc[idxs_shuffled]

        # pass responsibility for labelling to attached oracle
        annotated_data = self._get_annotations(shuffled_sample_unlabelled)

        _select = lambda label : annotated_data[FAIRNESS][FAIRNESS][lambda x : x == label]
        annotated_data_fair   = _select(FAIR)
        annotated_data_unfair = _select(UNFAIR)

        _old_len_fair   = len(self.dataset.training_data_fair)
        _old_len_unfair = len(self.dataset.training_data_unfair)

        # update training set
        assert len(set(annotated_data.index).intersection(set(self.dataset.training_data.index))) == 0
        
        self.dataset.set_training_data(self.dataset.training_data.index.union(annotated_data.index))

        _m =  'added annotations:\n'
        _m += '\t{} fair instances   + {} annotated "fair" instances   = {}  updated fair instance count\n'
        _m += '\t{} unfair instances + {} annotated "unfair" instances = {}  updated unfair instance count\n'
        self.logger.debug(_m.format(
            _old_len_fair,   len(annotated_data_fair),   len(self.dataset.training_data_fair),
            _old_len_unfair, len(annotated_data_unfair), len(self.dataset.training_data_unfair),
            verbosity = Verbosity.CHATTY
        ))

    def _annotate_and_retrain(self):
        self.logger.debug("Retraining model with new data")
            
        ########################################### train_model
        """Train model on the given training_data
        Tune with the validation_data
        Evaluate accuracy with the evaluation_data
        """
        # UPDATE OUR DATA AND (RE)TRAIN MODEL WITH NEWLY ANNOTATED DATA
        # vocab_size = create_features() # TODO: replace this method
        # TODO: custom labels
        self.logger.debug(f'Will train with learning_rate={self.config_manager.learning_rate}', verbosity=Verbosity.BASE)
        # epochs training

        model_path = self._train_and_evaluate(
            training_idxs=self.dataset.training_data.index,
            test_idxs=self.dataset.evaluation_data.index
        )
        ###########################################
        self.classifier.load_state_dict(torch.load(model_path))

        accuracies = self.classifier.evaluate_model(self.dataset.evaluation_data.index)
        self.logger.info(f"[fscore, auc] = {accuracies}")
        self.logger.info(f"Model saved to:  {model_path}")


    def _fill_training_data(self):
        # lets create our first training data! 
        self.logger.debug("Adding to initial training data...")

        idxs = self.dataset_manager.shuffle(data.index)
        needed = self.min_training_items - training_count
        data = data[:needed]
        # print(str(needed)+" more annotations needed")

        data = self._get_annotations(data)

        fair = []
        unfair = []

        for item in data:
            label = item[2]
            if label == "1":
                fair.append(item)
            elif label == "0":
                unfair.append(item)

        # append training data
        self.append_data(self.training_data_unfair, fair)
        self.append_data(self.training_data_fair, unfair)

    def _fill_evaluation_data(self):
        #Keep adding to evaluation data first
        self.logger.debug("Adding to evaluation data...")

        shuffle(data)
        needed = self.min_evaluation_items - evaluation_count
        data = data[:needed]
        self.logger.debug(f'{needed} more annotations needed')

        data = self._get_annotations(data) 
        
        fair = []
        unfair = []

        for item in data:
            label = item[2]    
            if label == "1":
                fair.append(item)
            elif label == "0":
                unfair.append(item)

        # append evaluation data 
        # TODO: implement append method relative to how data is being stored
        self.append_data(self.evaluation_data_fair, fair)
        self.append_data(self.evaluation_data_unfair, unfair)

    def _get_annotations(self, unlabelled_data: pd.DataFrame) -> pd.DataFrame:
        if self.config_manager.has_human_in_the_loop:
            return self._get_annotations_human(unlabelled_data)
        return self._get_annotations_auto(unlabelled_data)

    def _get_annotations_auto(self, unlabelled_data: pd.DataFrame) -> pd.DataFrame:
        # Nothing to do
        return unlabelled_data

    def _get_annotations_human(self) -> pd.DataFrame:
        # Get annotation data from annotator via command line
        raise NotImplementedError

if __name__=="__main__":
    main()