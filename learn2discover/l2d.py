import os
import sys
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from plotly import io
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from utils.reporter import get_output_dir
from configs.config_manager import ConfigManager

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
current_path = os.path.join(root_path, 'learn2discover')
lib_path = os.path.join(root_path,'lib')

for path in [root_path, current_path, lib_path]:
    if path not in sys.path:
        sys.path.append(path)
sys.path.index(current_path)
sys.path.index(lib_path)

from configs.config_manager import ConfigManager
from utils.reporter import Reporter
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

            self.reporter = Reporter()

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
        numerical_data = self.dataset.get_tensors_of_type(VarType.NUMERICAL)
        self.classifier = L2DClassifier(numerical_data.shape[1])
        self.classifier.attach(self.reporter)

        while not self.stopping_criterion():
            self.stopping_criterion.report()
            self.active_learning_loop()
        
        self.logger.info('Stopping criterion reached. Beginning evaluation...', verbosity=Verbosity.BASE)

        test_idxs_shuffled = self.dataset_manager.shuffle(self.dataset.test_data.index)
        result = self.classifier.evaluate_model(test_idxs_shuffled, final_report=True)
        fscore = result['f']
        auc    = result['auc']
        labels = result['y']
        y_pred  = result['y_pred']
        _m =  'RESULTS:\n'
        _m += 'Confusion Matrix: {}\n'.format(confusion_matrix(labels, y_pred))
        _m += f'Test Loss = {result["loss"]}\n'
        _m += f'Accuracy  = {result["acc"]}\n'
        _m += f'fscore    = {fscore}\n'
        _m += f'AUC       = {result["auc"]}\n'
        _m += f'Precision = {result["precision"]}\n'
        _m += f'Recall    = {result["recall"]}\n'
        self.logger.info(_m)
        self.logger.info(f'Classification Report: \n{classification_report(labels, y_pred)}')

        # Plot and save confusion matrix and final stats
        model_path = self.classifier.save_model(fscore, auc)
        self.logger.info(f"Model saved to:  {model_path}")
        
        self.reporter.report()

        # TODO: extract
        profile = get_output_dir()
        confusion_matrix_path = os.path.join(ConfigManager.get_instance().report_path, profile, "confusion_matrix.png")
        ConfusionMatrixDisplay.from_predictions(labels, y_pred)
        plt.savefig(confusion_matrix_path)
        self.logger.debug(f'Writing to "{confusion_matrix_path}"', verbosity=Verbosity.BASE)

        # Stats table
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Accuracy', 'F-score', 'AUC', 'Precision', 'Recall']),
            cells=dict(
                values=[
                    [round(result["acc"], 6)], 
                    [round(fscore, 6)], 
                    [round(result["auc"], 6)], 
                    [round(result["precision"], 6)], 
                    [round(result["recall"], 6)]]
                ))
            ])
        stats_path = os.path.join(ConfigManager.get_instance().report_path, profile, "stats.png")
        io.write_image(fig, stats_path)
        self.logger.debug(f'Writing to "{stats_path}"', verbosity=Verbosity.BASE)

    def active_learning_loop(self):
        """
        Perform the logic of the active learning loop
        """
        ##### TAKE SAMPLE OF UNLABELLED DATA AND PREDICT THE LABELS #####
        self.classifier.eval()  # stop training in order to query single samples

        sampled_idxs = self.dataset_manager.choose_random_unlabelled(self.dataset.unlabelled_data.index)
        random_items = self.dataset.unlabelled_data.loc[sampled_idxs]

        _m = 'run(): selected random sample of {} unlabelled instances'
        self.logger.debug(_m.format(len(sampled_idxs)), verbosity=Verbosity.CHATTY)
        self.logger.debug(f'First 5 sampled: \n{random_items[:5]}', verbosity=Verbosity.CHATTY)

        ##### APPLY QUERY STRATEGY #####
        # Sample unlabelled items based on query strategy
        sample_unlabelled = self.query_strategy.query(self.classifier, random_items)

        idxs_shuffled = self.dataset_manager.shuffle(sample_unlabelled.index)
        shuffled_sample_unlabelled = sample_unlabelled.loc[idxs_shuffled]

        ##### QUERY ORACLE FOR TRUE LABELS FOR THE INSTANCES #####
        # pass responsibility for labelling to attached oracle
        annotated_data = self._get_annotations(shuffled_sample_unlabelled)

        ##### ADD ANNOTATED INSTANCES TO THE SET OF LABELLED DATA #####
        self._update_training_data(annotated_data)

        ##### TRAIN MODEL USING SET OF CURRENTLY LABELLED DATA #####
        self.classifier.train()  # stop querying the model and continue training
        train_idxs_shuffled = self.dataset_manager.shuffle(self.dataset.training_data.index)
        self.classifier.fit(train_idxs_shuffled)

    def _update_training_data(self, annotated_data: pd.DataFrame) -> None:
        FAIRNESS = ParamType.FAIRNESS.value
        FAIR     = Label.FAIR.value
        UNFAIR   = Label.UNFAIR.value

        _select = lambda label : annotated_data[FAIRNESS][FAIRNESS][lambda x : x == label]
        annotated_data_fair   = _select(FAIR)
        annotated_data_unfair = _select(UNFAIR)

        _old_len_fair   = len(self.dataset.training_data.fair)
        _old_len_unfair = len(self.dataset.training_data.unfair)
        _m =  'will add annotations:\n'
        _m += '\t{} fair instances   + {} annotated "fair" instances   = {}  updated fair instance count\n'
        _m += '\t{} unfair instances + {} annotated "unfair" instances = {}  updated unfair instance count\n'
        self.logger.debug(_m.format(
            _old_len_fair,   len(annotated_data_fair),   _old_len_fair + len(annotated_data_fair),
            _old_len_unfair, len(annotated_data_unfair), _old_len_unfair + len(annotated_data_unfair),
            verbosity = Verbosity.CHATTY
        ))
        self.dataset.set_training_data(self.dataset.training_data.index.union(annotated_data.index))

    def _get_annotations(self, unlabelled_data: pd.DataFrame) -> pd.DataFrame:
        if self.config_manager.has_human_in_the_loop:
            annotated = self._get_annotations_human(unlabelled_data)
        else:
            annotated = self._get_annotations_auto(unlabelled_data)
        assert len(set(annotated.index).intersection(set(self.dataset.training_data.index))) == 0
        return annotated

    def _get_annotations_auto(self, unlabelled_data: pd.DataFrame) -> pd.DataFrame:
        # Nothing to do
        return unlabelled_data

    def _get_annotations_human(self) -> pd.DataFrame:
        # Get annotation data from annotator via command line
        raise NotImplementedError

if __name__=="__main__":
    main()