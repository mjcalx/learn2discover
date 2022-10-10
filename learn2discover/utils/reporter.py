import os
import hashlib
import json

from pathlib import Path

from utils.history import History
from utils.observer import Observer, Subject
from utils.logging_utils import Verbosity
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory

class Reporter(Observer):
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.history = History.get_instance()

    def update(self, subject: Subject):
        iter, loss, annotations, accuracy, confidence = subject.data

        self.history.concat(
            iter, loss, annotations, accuracy, confidence
        )


    def report(self) -> None:
        cfg = ConfigManager.get_instance()

        # Key output directory by hyperparameter values
        profile = get_output_dir()
        Path(cfg.report_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(cfg.report_path,profile)).mkdir(parents=True, exist_ok=True)
        
        # Write hyperparameters
        profile_path = os.path.join(cfg.report_path, profile, 'hyperparams.json')
        self.logger.debug(f'Writing to "{profile_path}"', verbosity=Verbosity.BASE)
        with open(profile_path, 'w') as f:
            json.dump(cfg.hyperparameters, f, indent=4)

        # Write reports
        csv_path = os.path.join(cfg.report_path, profile, "raw_data.csv")
        self.history.data.to_csv(csv_path, index=False)
        self.logger.debug(f'Writing to "{csv_path}"', verbosity=Verbosity.BASE)

        # Plot data to html file
        html_filename = os.path.join(cfg.report_path, profile, "results.html")
        self.history.plot_history(html_filename)
        self.logger.debug(f'Writing to "{html_filename}"', verbosity=Verbosity.BASE)

# TODO: extract
def get_output_dir():
    cfg = ConfigManager.get_instance()
    return hashlib.sha1(str.encode(str(cfg.hyperparameters))).hexdigest()[:5]