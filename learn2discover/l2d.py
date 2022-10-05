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
from utils.logging_utils import LoggingUtils
from loggers.logger_factory import LoggerFactory
from data.schema import VarType
from data.dataset_manager import DatasetManager
from data.data_classes import ParamType, Label
from oracle.query_strategies.query_strategy_factory import QueryStrategyFactory
from oracle.l2d_classifier import L2DClassifier

# FAIRNESS = ParamType.FAIRNESS.value
# FAIR = Label.FAIR.value
# UNFAIR = Label.UNFAIR.value

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
            
            self.test_fraction = self.config_manager.test_fraction
            
            _uf = self.config_manager.unlabelled_fraction
            self.logger.debug(f'UNLAB {_uf}')
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
        if self.dataset.evaluation_count <  self.min_evaluation_items:
            self._fill_evaluation_data()
         
        elif self.dataset.training_count < self.min_training_items:
            self._fill_training_data()

        else:
            self._learn()

        if self.dataset.training_count > self.min_training_items:
            self._annotate_and_retrain()

    # TODO ????
    def get_outliers(self, training_data, unlabeled_data, number=10):
        """Get outliers from unlabeled data in training data
        Returns number outliers
        
        An outlier is defined as the percent of words in an item in 
        unlabeled_data that do not exist in training_data
        """
        outliers = []

        total_feature_counts = defaultdict(lambda: 0)
        
        for item in training_data:
            text = item[1]
            features = text.split()

            for feature in features:
                total_feature_counts[feature] += 1
                    
        while(len(outliers) < number):
            top_outlier = []
            top_match = float("inf")

            for item in unlabeled_data:
                textid = item[0]
                if textid in self.already_labeled:
                    continue

                text = item[1]
                features = text.split()
                total_matches = 1 # start at 1 for slight smoothing 
                for feature in features:
                    if feature in total_feature_counts:
                        total_matches += total_feature_counts[feature]

                ave_matches = total_matches / len(features)
                if ave_matches < top_match:
                    top_match = ave_matches
                    top_outlier = item

            # add this outlier to list and update what is 'labeled', 
            # assuming this new outlier will get a label
            top_outlier[3] = "outlier"
            outliers.append(top_outlier)
            text = top_outlier[1]
            features = text.split()
            for feature in features:
                total_feature_counts[feature] += 1
        return outliers

    class SplitDataset:
        def __init__(self):
            pass

    def _learn(self):
        # Train new model with current training data
        embedding_size_function = lambda num_categories_per_col: [(n, min(50, (n+1)//2)) for n in num_categories_per_col]
        numerical_data = self.dataset.get_tensors_of_type(VarType.NUMERICAL)

        embedding_sizes = self.dataset.categorical_embedding_sizes
        numerical_data = self.dataset.get_tensors_of_type(VarType.NUMERICAL)

        self.classifier = L2DClassifier(embedding_sizes, numerical_data.shape[1])
        
        # Training
        self.logger.debug(self.dataset.training_data.index)
        self.classifier.fit(self.dataset.training_data.index)

        # Evaluation
        fscore, auc = self.classifier.evaluate_model(self.dataset.evaluation_data.index)
        model_path = self.classifier.save_model(fscore, auc)

        self.logger.debug("Sampling via Active Learning:\n")
        self.classifier.load_state_dict(torch.load(model_path))
        
        # stop training in order to query single samples
        self.classifier.eval()
        
        # get 100 items per iteration with the following breakdown of strategies:
        random_idxs = self.dataset_manager.choose_random_unlabelled(unlabelled_idxs)
        _m = 'Learn2Discover.run(): selected random sample of {} unlabelled instances'
        self.logger.debug(_m.format(len(random_idxs)))

        random_items = self.unlabelled_data.loc[random_idxs]
        self.logger.debug(f'First 5 sampled: \n{random_items[:5]}', verbosity=0)
        
        sample_unlabelled = self.query_strategy.query(self.classifier, random_items)
        # stop using existing model for queries and continue training
        # self.classifier.train()
        ########TODO GET OUTLIERS??###########
        # outliers = self.get_outliers(training_data+random_items+low_confidences, data, number=10)

        # sampled_data = random_items + low_confidences + outliers
        # shuffle(sampled_data)
        shuffled_sample_unlabelled = self.dataset_manager.shuffle(sample_unlabelled)
        ###################
        df = self.dataset_manager.data.all_columns()
        annotated_data = self._get_annotations(sample_unlabelled)
        self.annotated_data_fair   = df.loc[_index_fn(annotated_data.index, FAIR)]
        self.annotated_data_unfair = df.loc[_index_fn(annotated_data.index, UNFAIR)]
        assert len(self.annotated_data_fair) + len(self.annotated_data_unfair) == len(annotated_data)

        _old_len_fair   = len(self.training_data_fair)
        _old_len_unfair = len(self.training_data_unfair)

        self.training_data_fair   = pd.concat([self.training_data_fair,   self.annotated_data_fair])
        self.training_data_unfair = pd.concat([self.training_data_unfair, self.annotated_data_unfair])

        _m =  'added annotations:\n'
        _m += '\t{} fair instances   + {} annotated "fair" instances   = {}  updated fair instance count\n'
        _m += '\t{} unfair instances + {} annotated "unfair" instances = {}  updated unfair instance count\n'
        self.logger.debug(_m.format(
            _old_len_fair,   len(self.annotated_data_fair),   len(self.training_data_fair),
            _old_len_unfair, len(self.annotated_data_unfair), len(self.training_data_unfair),
            verbosity = 2
        ))

        self._update_training_count()

    def _annotate_and_retrain(self):
        self.logger.debug("\nRetraining model with new data")
            
        # UPDATE OUR DATA AND (RE)TRAIN MODEL WITH NEWLY ANNOTATED DATA
        training_data = self.dataset_manager.load_data(self.training_data_fair) + self.dataset_manager.load_data(self.training_data_unfair)
        _update_training_count()

        evaluation_data = self.dataset_manager.load_data(self.evaluation_data_fair) + self.dataset_manager.load_data(self.evaluation_data_unfair)
        _update_evaluation_count()

        vocab_size = create_features() # TODO: replace this method
        ########################################### train_model
        """Train model on the given training_data
        Tune with the validation_data
        Evaluate accuracy with the evaluation_data
        """
        assert self.training_data is not None
        assert self.evaluation_data is not None

        # TODO: custom labels
        self.logger.debug(f'Will train with learning_rate={self.config_manager.learning_rate} ')
        # epochs training

        self.classifier.fit(x=[training_data_fair, training_data_unfair], y=[labels])
        fscore, auc = self.classifier.evaluate_model(self.evaluation_data)
        model_path = model.save_model(fscore, auc)
        ###########################################
        self.classifier.load_state_dict(torch.load(model_path))

        accuracies = self.evaluate_model(model, evaluation_data)
        self.logger.info(f"[fscore, auc] = {accuracies}")
        self.logger.info(f"Model saved to:  {model_path}")


    def _fill_training_data(self):
        # lets create our first training data! 
        self.logger.debug("Adding to initial training data...")

        self.dataset_manager.shuffle(data)
        needed = self.min_training_items - training_count
        data = data[:needed]
        print(str(needed)+" more annotations needed")

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