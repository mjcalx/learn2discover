import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from configs.config_manager import ConfigManager
from data.dataset_manager import DatasetManager
from data.data_classes import ParamType, Label
from loggers.logger_factory import LoggerFactory


class L2DClassifier(nn.Module):
    def __init__(self, embedding_size, num_numerical_cols, layers):
        super(L2DClassifier, self).__init__()
        cfg = ConfigManager.get_instance()
        self.datamgr = DatasetManager.get_instance()
        self.logger = LoggerFactory.get_logger(__class__.__name__)

        self.epochs = cfg.epochs
        self.learning_rate = cfg.learning_rate
        self.selections_per_epoch = cfg.selections_per_epoch
        self.num_layers = cfg.num_layers
        self.dropout_rate = cfg.dropout_rate

        self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(self.dropout_rate)
        self.batch_norm = nn.BatchNorm1d(num_numerical_cols)
        self.layers = layers

        self.stack = None
        self._build(len_input=len(self.datamgr.attributes.inputs))

        #todo what are the parameters?
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss_function = nn.NLLLoss()
        self.metrics = ['fscore', 'auc']
        
    # def parameters(self):
    #     pass

    def forward(self, x_categorical, x_numerical):
        # Define how data is passed through the model
        embeddings = []
        for i,e in enumerate(self.embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings,1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)

        return x

        # hidden1 = self.linear1(feature_vec).clamp(min=0) # ReLU
        # output = self.linear2(hidden1)
        # return F.log_softmax(output, dim=1)

    def _build(self, len_input: int):
        i = 128
        all_layers = []
        # self.layer_linear_final = nn.Linear(i, len(Label))

        for i in self.layers:
            all_layers.append(nn.Linear(len_input, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(self.dropout_rate))
            len_input = i
        all_layers.append(nn.Linear(self.layers[-1], len(Label)))
        # layers.append(nn.Linear(i, len(Label)))

        self.stack = nn.Sequential(*all_layers)

    def fit(self, training_data: pd.DataFrame, epochs=None):
        epochs = epochs if epochs is not None else self.epochs
        
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
                    
            # train our model
            print(epoch_data)
            print(epoch_data[FAIRNESS])
            lambda x: self.datamgr.feature_vector(x)
            for item in epoch_data:
                features = item[1].split()
                label = int(item[2])

                self.zero_grad() 
                feature_vec = self.make_feature_vector(features, self.feature_index)
                target = torch.LongTensor([int(label)])

                log_probs = model(feature_vec)

                # compute loss function, do backward pass, and update the gradient
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()    

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

    def save_model(self):
        fscore = round(fscore,3)
        auc = round(auc,3)

        # save model to path that is alphanumeric and includes number of items and accuracies in filename
        timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
        training_size = "_"+str(len(self.training_data))
        accuracies = str(fscore)+"_"+str(auc)
                        
        model_path = "models/"+timestamp+accuracies+training_size+".params"

        torch.save(model.state_dict(), model_path)
        return model_path
