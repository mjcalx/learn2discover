import os
import sys
import pandas as pd
from typing import List

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)) )
l2d_path = os.path.join(root_path, 'learn2discover')
try:
    sys.path.index(l2d_path)
except ValueError:
    sys.path.append(l2d_path)

from configs.config_manager import ConfigManager
from data.schema import Schema
from data.loader import Loader
from data.data_classes import DataAttributes, FileData, DataInstance, Outcome, Label
from oracle.dataset_manager import DatasetManager
from collections import OrderedDict
from mock_oracle import set_labels

def _read_compas_data(filepath: str) -> FileData:
    """
    Reads a COMPAS csv data file
    """
    return parse_data(filepath, INPUT_ATTRIBUTES, OUTPUT_ATTRIBUTES)

def _determine_compas_outcomes(outputs: pd.DataFrame):
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
    schema = datamgr.schema
    data   = datamgr.data
    assert datamgr.schema is not None

    OUTPUT_ATTRIBUTES = ['DisplayText', 'RawScore', 'DecileScore', 'ScoreText', 'AssessmentType', 'IsCompleted',
                        'IsDeleted']
    INPUT_ATTRIBUTES = [i for i in schema.keys() if i not in set(OUTPUT_ATTRIBUTES)]
    SENSITIVE = ["Sex_Code_Text", "Ethnic_Code_Text"]

    # parse_data():
    attributes = DataAttributes(INPUT_ATTRIBUTES, OUTPUT_ATTRIBUTES)

    datamgr.parse_data_instances(attributes)

    datamgr.outcomes = _determine_compas_outcomes(datamgr.Y)
    assert len(datamgr.outcomes) == len(datamgr.X)
    datamgr.fairness_labels = set_labels(datamgr, SENSITIVE)
    label_count = lambda label : len(datamgr.fairness_labels[lambda x : x == label])
    fair_count   = label_count(Label.FAIR)
    unfair_count = label_count(Label.UNFAIR)
    print(f"Fair instances: {fair_count}. Unfair instances: {unfair_count}")
    print(
        f"Fair instances %: {fair_count / len(datamgr.data)}%. Unfair instances %: {unfair_count / len(datamgr.data)}%")
