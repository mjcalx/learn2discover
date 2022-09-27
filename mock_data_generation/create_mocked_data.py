import csv
from typing import List, Callable

import numpy as np

from data_classes import DataAttributes, DataInstance, FileData, TestOutcome
from mock_data_generation.mock_oracle import set_mock_labels


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


def _parse_data(filepath: str, input_attributes: List[str], output_attributes: List[str]) -> FileData:
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
    instances = _parse_instances(rows, attributes)
    # Get the file's original name from the filepath (excluding the .csv extension)
    filename = filepath.split("/")[-1][:-4]

    return FileData(attributes, instances, filename)


def get_mock_data(filepath: str, input_attributes: List[str], output_attributes: List[str],
                  sensitive_attributes: List[str], outcome_func: Callable[[DataInstance], TestOutcome]) -> FileData:
    """Generates mocked fairness labels for a specified raw dataset.

    Parameters:
        filepath (str): Path to the file to read

        input_attributes (List[str]): A list of expected SUT input attributes that the file should contain

        output_attributes (List[str]): A list of expected SUT output attributes that the file should contain

        sensitive_attributes (List[str]): A list of attributes from the input_attributes to consider as "sensitive",
            i.e. contributing to a decision on fairness

        outcome_func (Callable[[DataInstance], TestOutcome]): A function unique to the specified dataset that
            determines whether a given data instance from the dataset passes or fails some goal or objective,
            e.g. to receive a loan or not receive a loan

    Returns:
        A FileData object with all data instances containing a fairness label
    """
    # Get the unlabelled data from the raw csv data file
    unlabelled_data = _parse_data(filepath, input_attributes, output_attributes)

    # Determine the "test outcome" for each data instance
    for instance in unlabelled_data.instances:
        instance.outcome = outcome_func(instance)

    # Apply mocked fairness labels to each data instance
    return set_mock_labels(unlabelled_data, sensitive_attributes)


def _filedata_is_labelled(file_data: FileData):
    """
    Checks whether each data instance in the file_data has been assigned both a TestOutcome and a FairnessLabel
    """
    for instance in file_data.instances:
        if instance.label is None or instance.outcome is None:
            return False

    return True


def create_l2d_input_csv(labelled_data: FileData):
    """Creates a new CSV file in the format expected by the active learning loop in Learn2Discover

    Parameters:
          labelled_data (FileData): Data from the raw data file where each data instance has been assigned a
          TestOutcome and a FairnessLabel
    """
    assert _filedata_is_labelled(labelled_data), "The file data has not been fully labelled with both test outcomes " \
                                                 "and fairness labels"

    TEST_OUTCOME = "TEST_OUTCOME"
    FAIRNESS_LABEL = "FAIRNESS_LABEL"
    csv_path = f"../datasets/mock_generated/{labelled_data.filename}-MOCK_LABELLING.csv"

    with open(csv_path, "w", newline="") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

        # Write the headers
        headers_array = np.concatenate((labelled_data.attributes.inputs, ([TEST_OUTCOME, FAIRNESS_LABEL])))
        filewriter.writerow(headers_array)

        # Write each data instance
        for instance in labelled_data.instances:
            instance_array = []
            for header in headers_array:
                if header == TEST_OUTCOME:
                    instance_array.append(instance.outcome.name)
                elif header == FAIRNESS_LABEL:
                    instance_array.append(instance.label.name)
                else:
                    instance_array.append(instance.inputs[header])
            filewriter.writerow(instance_array)
    csvfile.close()
