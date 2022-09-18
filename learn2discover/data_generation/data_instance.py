import csv
from enum import Enum
from typing import List, Optional


class DataAttributes:
    """
    Stores information about the data attributes for a dataset
    """

    def __init__(self, inputs: List[str], outputs: List[str]):
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs


class Label(Enum):
    """
    A binary label with values "Fair" and "Unfair"
    """
    FAIR = True
    UNFAIR = False


class DataInstance:
    """
    A class for a single data instance. Should contain:
        - SUT input values
        - SUT output values
        - Fairness label
    """

    def __init__(self, inputs: List[str], outputs: List[str], label: Optional[Label] = None):
        self._inputs = inputs
        self._outputs = outputs
        self._label: Label = label

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: Label):
        self._label = label

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: List[str]):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[str]):
        self._outputs = outputs


def test_csv():
    """
    Basic tests for csv file reading
    """
    filepath = "../../datasets/original/COMPAS/compas-scores-raw.csv"

    headers = []
    rows = []

    with open(filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        headers = next(csv_reader)

        for row in csv_reader:
            rows.append(row)

    print("Header names are:")
    for header in headers:
        print(f"\t{header}")

    print("First 5 rows are:")
    for row in rows[:5]:
        print(row)
        # for col in row:
        #     print("%10s"%col, end=" "),
        print("\n")


def test_class():
    """
    Basic tests for data class initializing
    """
    attributes = DataAttributes(["attribute 1"], ["attribute 2"])
    data = DataInstance(["input 1"], ["output 1"])
    data.label = Label.FAIR
    print(f"{attributes.inputs}, {attributes.outputs}")
    print(f"{data.inputs}, {data.outputs}, {data.label}")


if __name__ == "__main__":
    # test_csv()
    test_class()
