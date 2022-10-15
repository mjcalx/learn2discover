import os
import hashlib
import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from utils.history import History
from utils.logging_utils import Verbosity
from utils.observer import Observer, Subject
from utils.summary import summary_plot, summary_dict
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory

class Reporter(Observer):
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.history = History.get_instance()

    def update(self, subject: Subject):
        iter, loss, annotations, accuracy, precision, confidence = subject.data

        self.history.concat(
            iter, loss, annotations, accuracy, precision, confidence
        )

    def report(self, result: Dict[str, object]) -> None:
        cfg = ConfigManager.get_instance()

        # Print results to console
        fscore = result['f']
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

        # Key output directory by configuration
        profile = Reporter.get_report_dir()
        Path(cfg.report_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(cfg.report_path,profile)).mkdir(parents=True, exist_ok=True)
        
        # Write run configuration
        profile_path = os.path.join(cfg.report_path, profile, 'configuration.json')
        self.logger.debug(f'Writing to "{profile_path}"', verbosity=Verbosity.BASE)
        with open(profile_path, 'w') as f:
            json.dump(Reporter._output_key(), f, indent=4)

        # Write reports
        csv_path = os.path.join(cfg.report_path, profile, "raw_data.csv")
        self.history.data.to_csv(csv_path, index=False)
        self.logger.debug(f'Writing to "{csv_path}"', verbosity=Verbosity.BASE)

        # Plot data to html file
        html_filename = os.path.join(cfg.report_path, profile, "results.html")
        summary_plot(csv_path, html_filename)
        self.logger.debug(f'Writing to "{html_filename}"', verbosity=Verbosity.BASE)

        # Plot and save confusion matrix and final stats
        confusion_matrix_path = os.path.join(cfg.report_path, profile, "confusion_matrix.png")
        ConfusionMatrixDisplay.from_predictions(labels, y_pred)
        plt.savefig(confusion_matrix_path)
        self.logger.debug(f'Writing to "{confusion_matrix_path}"', verbosity=Verbosity.BASE)

        # Write stats files
        stats_path = os.path.join(cfg.report_path, profile, "stats.json")
        summary_dict(stats_path, {k: result[k] for k in ('f', 'auc', 'acc', 'precision', 'recall')})
        self.logger.debug(f'Writing to "{stats_path}"', verbosity=Verbosity.BASE)

        self.history.reset()

    @staticmethod
    def get_report_dir() -> str:
        return hashlib.sha1(str.encode(str(Reporter._output_key()))).hexdigest()[:5]

    @staticmethod
    def _output_key() -> Dict[str, object]:
        cfg = ConfigManager.get_instance()
        sections = ["model_hyperparameters", "training_settings", "dataset_settings"]
        configs = {k:cfg.configs[k] for k in cfg.configs.keys() if k in sections}
        return configs


