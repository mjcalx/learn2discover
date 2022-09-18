import csv
from typing import List

from data_classes import DataAttributes, DataInstance, FileData


def _headers_are_valid(headers: List[str], input_attributes: List[str], output_attributes: List[str]) -> bool:
    """
    Checks whether the headers received match the hardcoded expected attributes
    """
    input_count = len(input_attributes)

    for i in range(len(headers)):
        if (i < input_count and headers[i] != input_attributes[i]) or (
                i >= input_count and headers[i] != output_attributes[i - input_count]):
            return False
    return True


def _parse_instances(row_data: List[List[str]], input_count: int) -> List[DataInstance]:
    """
    Parses raw row data to a list of DataInstances
    """
    instances: List[DataInstance] = []

    for i in range(len(row_data)):
        inputs = row_data[i][:input_count]
        outputs = row_data[i][input_count:]

        instances.append(DataInstance(inputs, outputs))

    return instances


def parse_data(filepath: str, input_attributes: List[str], output_attributes: List[str]) -> FileData:
    """Reads a csv data file and parses it into relevant data types for model to use.

    Parameters:
        filepath (str): Path to the file to read

        input_attributes (List[str]): A list of expected SUT input attributes that the file should contain

        output_attributes (List[str]): A list of expected SUT output attributes that the file should contain

    Returns:
        A FileData object containing information on the file's attributes and data instances
    """
    headers: List[str]
    rows = []

    with open(filepath, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        headers = next(csv_reader)

        for row in csv_reader:
            rows.append(row)

    # Check that the headers are as expected
    assert _headers_are_valid(
        headers, input_attributes, output_attributes), f"Headers are:\n{headers}\nbut expected:\n\tinputs:" \
                                                       f" {input_attributes}\n\toutputs:" \
                                                       f" {output_attributes}"

    attributes = DataAttributes(input_attributes, output_attributes)
    instances = _parse_instances(rows, len(attributes.inputs))

    return FileData(attributes, instances)
