import os
import sys
import traceback
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

def main():
    learn_to_discover = Learn2Discover()
    learn_to_discover.run()

class Learn2Discover:
    """
    Learn2Discover
    """
    def __init__(self, workspace_dir=os.getcwd(), human=False):
        self.config_manager = ConfigManager(workspace_dir)
        LoggingUtils.get_instance().debug('Loaded Configurations.')
        try:
            self.logger = LoggerFactory.get_logger(__class__.__name__)
            self.logger.debug('Loaded Logger.')
            
            self.dataset_manager = DatasetManager()
            self.logger.debug('Loaded DatasetManager.')
            
            self.query_strategies = [QueryStrategyFactory().get_strategy(t) for t in self.config_manager.query_strategies]

            # Constants
            self.test_fraction = 0.2
            return
            self.min_evaluation_items = 200
            self.min_training_items = 50
            self.epochs = 10
            self.selections_per_epoch = 40

            self.already_labeled = {} # tracking what is already labeled
            self.feature_index = {} # feature mapping for one-hot encoding

            self.evaluation_count = len(self.evaluation_data)
            self.training_count = len(self.training_data)

            # Human-specific params
            if human: 
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
    
    def train_model(self, num_labels=2, vocab_size=0):
        """Train model on the given training_data
        Tune with the validation_data
        Evaluate accuracy with the evaluation_data
        """

        model = SimpleTextClassifier(num_labels, vocab_size)
        
        # TODO: custom labels
        label_to_ix = {"fair": 0, "unfair": 1} 

        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # epochs training
        for epoch in range(self.epochs):
            print("Epoch: "+str(epoch))
            current = 0

            # make a subset of data to use in this epoch
            # with an equal number of items from each label

            shuffle(self.training_data) #randomize the order of the training data        
            fair = [row for row in self.training_data if '1' == row[2]]
            unfair = [row for row in self.training_data if '0' == row[2]]
            
            epoch_data = fair[:self.selections_per_epoch]
            epoch_data += unfair[:self.selections_per_epoch]
            shuffle(epoch_data) 
                    
            # train our model
            for item in epoch_data:
                features = item[1].split()
                label = int(item[2])

                model.zero_grad() 

                feature_vec = self.make_feature_vector(features, self.feature_index)
                target = torch.LongTensor([int(label)])

                log_probs = model(feature_vec)

                # compute loss function, do backward pass, and update the gradient
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()    

        fscore, auc = self.evaluate_model(model, self.evaluation_data)
        fscore = round(fscore,3)
        auc = round(auc,3)

        # save model to path that is alphanumeric and includes number of items and accuracies in filename
        timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
        training_size = "_"+str(len(self.training_data))
        accuracies = str(fscore)+"_"+str(auc)
                        
        model_path = "models/"+timestamp+accuracies+training_size+".params"

        torch.save(model.state_dict(), model_path)
        return model_path

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


    def get_random_items(self, unlabeled_data, number = 10):
        shuffle(unlabeled_data)

        random_items = []
        for item in unlabeled_data:
            textid = item[0]
            if textid in self.already_labeled:
                continue
            item[3] = "random_remaining"
            random_items.append(item)
            if len(random_items) >= number:
                break

        return random_items


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
        


    def evaluate_model(self, model, evaluation_data):
        """Evaluate the model on the held-out evaluation data
        Return the f-value for disaster-related and the AUC
        """

        related_confs = [] # related items and their confidence of being related
        not_related_confs = [] # not related items and their confidence of being _related_

        true_pos = 0.0 # true positives, etc 
        false_pos = 0.0
        false_neg = 0.0

        with torch.no_grad():
            for item in evaluation_data:
                _, text, label, _, _, = item

                feature_vector = self.make_feature_vector(text.split(), self.feature_index)
                log_probs = model(feature_vector)

                # get confidence that item is disaster-related
                prob_fair = math.exp(log_probs.data.tolist()[0][1]) 

                if(label == "1"):
                    # true label is disaster related
                    related_confs.append(prob_fair)
                    if prob_fair > 0.5:
                        true_pos += 1.0
                    else:
                        false_neg += 1.0
                else:
                    # not disaster-related
                    not_related_confs.append(prob_fair)
                    if prob_fair > 0.5:
                        false_pos += 1.0

        # Get FScore
        if true_pos == 0.0:
            fscore = 0.0
        else:
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            fscore = (2 * precision * recall) / (precision + recall)

        # GET AUC
        not_related_confs.sort()
        total_greater = 0 # count of how many total have higher confidence
        for conf in related_confs:
            for conf2 in not_related_confs:
                if conf < conf2:
                    break
                else:                  
                    total_greater += 1


        denom = len(not_related_confs) * len(related_confs) 
        auc = total_greater / denom

        return[fscore, auc]

    def run(self):

        # TODO: need to split training/evaluation data within the dataset manager
        # Alternatively, can randomly split a full dataset here based on ratios
        data = self.dataset_manager.data
        tensor_data = self.dataset_manager.tensor_data
        self.logger.debug(f'{len(data)} instances loaded.')

        train_idxs, test_idxs = self.dataset_manager.split_dataset(self.test_fraction)
        training_data = tensor_data.loc(train_idxs)
        test_data = tensor_data.loc(test_idxs)

        print(data.fairness_labels)
        data.fairness_labels[Label.FAIR]
        self.training_data_fair = self.dataset_manager.get_fair(training_data)
        self.training_data_unfair = self.dataset_manager.get_unfair(training_data)

        self.evaluation_data = self.dataset_manager.load_data(dataset)
        self.evaluation_data_fair = self.dataset_manager.get_fair(evaluation_data)
        self.evaluation_data_unfair = self.dataset_manager.get_unfair(evaluation_data)
        
        self.unlabelled_data = self.dataset_manager.load_data(dataset) # TODO: remove this eventually

        if self.evaluation_count <  self.min_evaluation_items:
            #Keep adding to evaluation data first
            self.logger.info("Creating evaluation data:")

            shuffle(data)
            needed = self.min_evaluation_items - evaluation_count
            data = data[:needed]
            self.logger.info(f'{needed} more annotations needed')

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

        elif training_count < self.min_training_items:
            # lets create our first training data! 
            print("Creating initial training data:\n")

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
            model_path = self.train_model(training_data, evaluation_data=evaluation_data)

            print("Sampling via Active Learning:\n")

            model = SimpleTextClassifier(2, vocab_size)
            model.load_state_dict(torch.load(model_path))

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
            self.append_data(self.training_data_fair, fair)
            self.append_data(self.training_data_unfair, unfair)
            

        if training_count > self.min_training_items:
            print("\nRetraining model with new data")
            
            # UPDATE OUR DATA AND (RE)TRAIN MODEL WITH NEWLY ANNOTATED DATA
            training_data = self.dataset_manager.load_data(self.training_data_fair) + self.dataset_manager.load_data(self.training_data_unfair)
            training_count = len(training_data)

            evaluation_data = self.dataset_manager.load_data(self.evaluation_data_fair) + self.dataset_manager.load_data(self.evaluation_data_unfair)
            evaluation_count = len(evaluation_data)

            vocab_size = create_features() # TODO: replace this method
            model_path = self.train_model(training_data, evaluation_data=evaluation_data, vocab_size=vocab_size)
            model = SimpleTextClassifier(2, vocab_size)
            model.load_state_dict(torch.load(model_path))

            accuracies = self.evaluate_model(model, evaluation_data)
            self.logger.info(f"[fscore, auc] = {accuracies}")
            self.logger.info(f"Model saved to:  {model_path}")

if __name__=="__main__":
    main()