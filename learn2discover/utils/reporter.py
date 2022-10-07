import os
import pandas as pd
import hashlib
import json

from enum import Enum
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt

from utils.observer import Observer, Subject
from utils.logging_utils import Verbosity
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory

class Report(Enum):
    VALIDATION_LOSS_PER_ITERATION = "validation-loss-per-iteration"
    TEST_LOSS_PER_ANNOTATION = "test-loss-per-annotation"

ITER_C = 'Iteration'
VLOSS_C = 'Validation Loss'
TLOSS_C = 'Test Loss'
ANNOT_C = 'Number of annotations'

class Reporter(Observer):
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__class__.__name__)

        self._vloss_per_iter = {ITER_C:[], VLOSS_C:[]}
        self._tloss_per_annotation = {ANNOT_C:[], TLOSS_C:[]}

        self._report_types = {
            Report.VALIDATION_LOSS_PER_ITERATION : self._vloss_per_iter,
            Report.TEST_LOSS_PER_ANNOTATION      : self._tloss_per_annotation,
        }

    def update(self, subject: Subject):
        key, report = subject.report

        if key == Report.VALIDATION_LOSS_PER_ITERATION:
            iteration, vloss = report
            self._report_types[key][ITER_C].append(iteration)
            self._report_types[key][VLOSS_C].append(vloss)

        if key == Report.TEST_LOSS_PER_ANNOTATION:
            annotation_count, tloss = report
            self._report_types[key][ANNOT_C].append(annotation_count)
            self._report_types[key][TLOSS_C].append(tloss)

    def report(self) -> None:
        cfg = ConfigManager.get_instance()

        # Key output directory by hyperparameter values
        profile = hashlib.sha1(str.encode(str(cfg.hyperparameters))).hexdigest()[:5]
        Path(cfg.report_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(cfg.report_path,profile)).mkdir(parents=True, exist_ok=True)
        
        # Write hyperparameters
        profile_path = os.path.join(cfg.report_path, profile, 'hyperparams.json')
        self.logger.debug(f'Writing to "{profile_path}"', verbosity=Verbosity.BASE)
        with open(profile_path, 'w') as f:
            json.dump(cfg.hyperparameters, f, indent=4)

        # Write reports
        for k in self._report_types.keys():
            csv_path = os.path.join(cfg.report_path, profile, k.value + ".csv")
            df = pd.DataFrame.from_dict(self._report_types[k])
            df.to_csv(csv_path, index=False)
            self.logger.debug(f'Writing to "{csv_path}"', verbosity=Verbosity.BASE)

            # # #TODO EXTRACT
            # plt.plot(self._training_loss_per_epoch[EPOCH_C], self._training_loss_per_epoch[TRAIN_LOSS_C])
            # plt.ylabel(TRAIN_LOSS_C)
            # plt.xlabel(EPOCH_C)
            # plt.savefig(os.path.join(out_path, k.value + ".png"))
            # plt.clf()