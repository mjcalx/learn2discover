import csv
from typing import List

from data.data_classes import DataAttributes, DataInstance, FileData

def _parse_instances(row_data: List[List[str]], attributes: DataAttributes) -> List[DataInstance]:
    """
    Parses raw row data to a list of DataInstances
    """
    input_count = len(attributes.inputs)
    output_count = len(attributes.outputs)
    instances: List[DataInstance] = []

    for row in row_data:
        inputs = {}
        outputs = {}
        for i in range(input_count):
            inputs[attributes.inputs[i]] = row[i]

        for i in range(input_count, input_count + output_count):
            outputs[attributes.outputs[i - input_count]] = row[i]

        instances.append(DataInstance(inputs, outputs))

    return instances


def parse_data(input_attributes: List[str], output_attributes: List[str]) -> FileData:
    """Reads a csv data file and parses it into relevant data types for model to use.

    Parameters:
        filepath (str): Path to the file to read

        input_attributes (List[str]): A list of expected SUT input attributes that the file should contain

        output_attributes (List[str]): A list of expected SUT output attributes that the file should contain

    Returns:
        A FileData object containing information on the file's attributes and data instances
    """
    attributes = DataAttributes(input_attributes, output_attributes)
    instances = _parse_instances(rows, attributes)

    return FileData(attributes, instances)
