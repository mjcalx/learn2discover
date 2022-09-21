import os
import sys
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
from data.data_classes import DataAttributes, FileData, DataInstance, Outcome
from collections import OrderedDict
from mock_oracle import set_labels
from read_data import parse_data

def _read_compas_data(filepath: str) -> FileData:
    """
    Reads a COMPAS csv data file
    """
    return parse_data(filepath, INPUT_ATTRIBUTES, OUTPUT_ATTRIBUTES)

def _determine_compas_outcomes(instances: List[DataInstance]):
    """
    Determines the outcome for each data instance based on its output values
    """
    for instance in instances:
        # TODO: Determine a reasonable way to identify which instances pass and fail
        if instance.outputs["ScoreText"] in ["High", "Medium"]:
            instance.outcome = Outcome.FAIL
        else:
            instance.outcome = Outcome.PASS

def get_compas_data(filepath: str, sensitive_attributes: List[str]) -> FileData:
    """
    Reads COMPAS csv data files, and applies a fairness label to its sensitive attributes
    """
    # TODO: Refactor to take output decider function as input parameter
    unlabelled_data = _read_compas_data(filepath)
    _determine_compas_outcomes(unlabelled_data.instances)
    return set_labels(unlabelled_data, sensitive_attributes)


if __name__ == "__main__":
    """
    A simple test example to show that everything is working
    """
    config = ConfigManager(os.getcwd())
    loader = Loader()
    schema, data = loader.load_data()

    OUTPUT_ATTRIBUTES = ['DisplayText', 'RawScore', 'DecileScore', 'ScoreText', 'AssessmentType', 'IsCompleted',
                        'IsDeleted']
    INPUT_ATTRIBUTES = [i for i in schema.keys() if i not in set(OUTPUT_ATTRIBUTES)]

    testpath = "../datasets/original/COMPAS/compas-scores-raw.csv"
    sample_sensitive_attributes = ["Sex_Code_Text", "Ethnic_Code_Text"]

    # Get the labelled data
    data = get_compas_data(testpath, sample_sensitive_attributes)

    # See how many instances are fair vs unfair
    fair_count = 0
    unfair_count = 0
    for instance in data.instances:
        if instance.label.value:
            fair_count += 1
        else:
            unfair_count += 1

    print(f"Fair instances: {fair_count}. Unfair instances: {unfair_count}")
    print(
        f"Fair instances %: {fair_count / len(data.instances)}%. Unfair instances %: {unfair_count / len(data.instances)}%")
