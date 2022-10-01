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
# from lib.pytorch_active_learning.active_learning_basics import SimpleTextClassifier
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
            self.logger.debug('Loaded DatasetManager.')
            
            self.query_strategies = [QueryStrategyFactory().get_strategy(t) for t in self.config_manager.query_strategies]
            
            self.test_fraction = self.config_manager.test_fraction
            self.min_evaluation_items = self.config_manager.min_evaluation_items
            self.min_training_items = self.config_manager.min_training_items

            self.training_data = None
            self.training_count = None
            self.evaluation_data = None
            self.evaluation_count = None

            # Constants

            self.already_labeled = {} # tracking what is already labeled
            self.feature_index = {}   # feature mapping for one-hot encoding

            # Human-specific params
            if self.config_manager.has_human_in_the_loop: 
                self.get_annotations = self.get_annotations_human
            else:
                self.get_annotations = self.get_annotations_auto

        except BaseException as e:
            self.logger.error(f'{e}')
            self.logger.error('Exiting...')
            print(traceback.format_exc())
            exit()
    
    def get_annotations_human(self, data, default_sampling_strategy="random"):
        # Get annotation data from annotator via command line
        # TODO: actually code this
        pass

    def get_annotations_auto(self, data, default_sampling_strategy="random"):
        # don't need to do anything here :)
        return data 

    def append_data(data, to_add):
        # TODO: merge two sets of input data here
        pass
    
    @staticmethod
    def make_feature_vector(features, feature_index):
        vec = torch.zeros(len(feature_index))
        for feature in features:
            if feature in feature_index:
                vec[feature_index[feature]] += 1
        return vec.view(1, -1)
    

    def get_low_conf_unlabeled(self, model, unlabeled_data, number=80, limit=10000):
        confidences = []
        if limit == -1: # we're predicting confidence on *everything* this will take a while
            print("Get confidences for unlabeled data (this might take a while)")
        else: 
            # only apply the model to a limited number of items
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]
        with torch.no_grad():
            for item in unlabeled_data:
                textid = item[0]
                if textid in self.already_labeled:
                    continue
                item[3] = "random_remaining"
                text = item[1]

                feature_vector = self.make_feature_vector(text.split(), self.feature_index)
                log_probs = model(feature_vector)

                # get confidence that it is related
                prob_related = math.exp(log_probs.data.tolist()[0][1]) 
                
                if prob_related < 0.5:
                    confidence = 1 - prob_related
                else:
                    confidence = prob_related 

                item[3] = "low confidence"
                item[4] = confidence
                confidences.append(item)

        confidences.sort(key=lambda x: x[4])
        return confidences[:number:]



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

    def run(self):
        FAIRNESS = ParamType.FAIRNESS.value
        FAIR = Label.FAIR.value
        UNFAIR = Label.UNFAIR.value

        train_idxs, test_idxs = self.dataset_manager.split_dataset(self.test_fraction)
        ftd_obj = self.dataset_manager.data
        df = self.dataset_manager.data.all_columns()
        
        tensor_data = self.dataset_manager.tensor_data
        self.logger.debug(f'{len(df)} instances loaded.')

        #TENSORS
        # training_data = tensor_data.loc(train_idxs)
        # test_data = tensor_data.loc(test_idxs)

        _index_fn = lambda index_obj, label : df.loc[index_obj][FAIRNESS][lambda x : x[FAIRNESS] == label].index
        self.training_data   = df.loc[train_idxs]
        self.evaluation_data = df.loc[test_idxs]
        self.training_data_fair   = df.loc[_index_fn(train_idxs, FAIR)]
        self.training_data_unfair = df.loc[_index_fn(train_idxs, UNFAIR)]
        self.evaluation_data_fair   = df.loc[_index_fn(test_idxs, FAIR)]
        self.evaluation_data_unfair = df.loc[_index_fn(test_idxs, UNFAIR)]
        self.evaluation_count = len(self.evaluation_data_fair) + len(self.evaluation_data_unfair)
        self.training_count = len(self.training_data_fair) + len(self.training_data_unfair)

        msg = 'NUM_INSTANCES: training_fair={} | training_unfair={} | evaluation_fair={} | evaluation_unfair={}'
        lengths = [len(d) for d in [self.training_data_fair, self.training_data_unfair, self.evaluation_data_fair, self.evaluation_data_unfair]]
        self.logger.debug(msg.format(*lengths))
        assert sum(lengths)==len(df)==len(self.training_data)+len(self.evaluation_data)
        
        self.unlabelled_data = None # self.dataset_manager.load_data() # TODO: remove this eventually -- what's this for?

        self.evaluation_count = len(self.evaluation_data)
        self.training_count = len(self.training_data)
        
        if self.evaluation_count <  self.min_evaluation_items:
            #Keep adding to evaluation data first
            self.logger.debug("Creating evaluation data:")

            shuffle(data)
            needed = self.min_evaluation_items - evaluation_count
            data = data[:needed]
            self.logger.debug(f'{needed} more annotations needed')

            data = self.get_annotations(data) 
            
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
            exit()

        elif self.training_count < self.min_training_items:
            # lets create our first training data! 
            self.logger.debug("Creating initial training data:\n")

            shuffle(data)
            needed = self.min_training_items - training_count
            data = data[:needed]
            print(str(needed)+" more annotations needed")

            data = self.get_annotations(data)

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
        else:
            # lets start Active Learning!! 

            # Train new model with current training data

            # CLASSIFIER
            _tensor_data = self.dataset_manager.tensor_data
            embedding_sizes = _tensor_data.categorical_embedding_sizes
            numerical_data = _tensor_data.get_tensors_of_type(VarType.NUMERICAL)
            self.classifier = L2DClassifier(embedding_sizes, numerical_data.shape[1])
            # CLASSIFIER
            
            # Training
            self.classifier.fit(train_idxs)

            # Evaluation
            fscore, auc = self.classifier.evaluate_model(test_idxs)
            model_path = self.classifier.save_model(fscore, auc)

            self.logger.debug("Sampling via Active Learning:\n")
            self.classifier.load_state_dict(torch.load(model_path))

            # get 100 items per iteration with the following breakdown of strategies:
            random_items = self.get_random_items(data, number=10)
            low_confidences = self.get_low_conf_unlabeled(model, data, number=80)
            outliers = self.get_outliers(training_data+random_items+low_confidences, data, number=10)

            sampled_data = random_items + low_confidences + outliers
            shuffle(sampled_data)
            
            sampled_data = self.get_annotations(sampled_data)
            fair = []
            unfair = []
            for item in sampled_data:
                label = item[2]
                if label == "1":
                    fair.append(item)
                elif label == "0":
                    unfair.append(item)

            # append training data
            # todo indexes instead?
            self.append_data(self.training_data_fair, fair)
            self.append_data(self.training_data_unfair, unfair)
            

        if training_count > self.min_training_items:
            self.logger.debug("\nRetraining model with new data")
            
            # UPDATE OUR DATA AND (RE)TRAIN MODEL WITH NEWLY ANNOTATED DATA
            training_data = self.dataset_manager.load_data(self.training_data_fair) + self.dataset_manager.load_data(self.training_data_unfair)
            training_count = len(training_data)

            evaluation_data = self.dataset_manager.load_data(self.evaluation_data_fair) + self.dataset_manager.load_data(self.evaluation_data_unfair)
            evaluation_count = len(evaluation_data)

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

if __name__=="__main__":
    main()