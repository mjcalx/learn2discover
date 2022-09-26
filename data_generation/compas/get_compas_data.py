import os
import sys
import pandas as pd
from typing import List

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
l2d_path = os.path.join(root_path, 'learn2discover')
try:
    sys.path.index(l2d_path)
except ValueError:
    sys.path.append(l2d_path)

from configs.config_manager import ConfigManager
from data.schema import Schema
from data.loader import Loader
from data.data_classes import DataAttributes, Outcome, Label
from data.dataset_manager import DatasetManager
from collections import OrderedDict
from compas_oracle import CompasOracle

ORACLE_TYPE = CompasOracle

def _determine_compas_outcomes(outputs: pd.DataFrame) -> pd.Series:
    """
    Determines the outcome for each data instance based on its output values
    """
    assign_outcome = lambda x : Outcome.FAIL if str(x["ScoreText"]) in ["High", "Medium", "nan"] else Outcome.PASS
    return outputs.apply(assign_outcome, axis=1)

if __name__ == "__main__":
    """
    A simple test example to show that everything is working
    """
    config  = ConfigManager(os.getcwd())
    datamgr = DatasetManager()
    oracle = ORACLE_TYPE()

    schema = datamgr.schema
    data   = datamgr.data
    assert datamgr.schema is not None


    OUTPUT_ATTRIBUTES = ['DisplayText', 'RawScore', 'DecileScore', 'ScoreText', 'AssessmentType', 'IsCompleted',
                        'IsDeleted']
    INPUT_ATTRIBUTES = [i for i in schema.keys() if i not in set(OUTPUT_ATTRIBUTES)]
    SENSITIVE = ["Sex_Code_Text", "Ethnic_Code_Text"]

    attributes = DataAttributes(INPUT_ATTRIBUTES, OUTPUT_ATTRIBUTES)
    datamgr.parse_data_instances(attributes)
    datamgr.outcomes = _determine_compas_outcomes(datamgr.Y)

    assert len(datamgr.outcomes) == len(datamgr.X)
    datamgr.fairness_labels = oracle.set_labels(SENSITIVE)