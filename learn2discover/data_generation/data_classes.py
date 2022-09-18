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


class FileData:
    """
    A class for storing a full file's data
    """

    def __init__(self, attributes: DataAttributes, instances: List[DataInstance]):
        self._attributes = attributes
        self._instances = instances

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes: DataAttributes):
        self._attributes = attributes

    @property
    def instances(self):
        return self._instances

    @instances.setter
    def instances(self, instances: List[DataInstance]):
        self._instances = instances


def _test_class():
    """
    Basic tests for data class initializing
    """
    attributes = DataAttributes(["attribute 1"], ["attribute 2"])
    data = DataInstance(["input 1"], ["output 1"])
    data.label = Label.FAIR

    print(f"{attributes.inputs}, {attributes.outputs}")
    if type(data.label) is Label:
        print(f"{data.inputs}, {data.outputs}, {data.label.name}/{data.label.value}")
    else:
        print(f"{data.inputs}, {data.outputs}, {data.label}")


if __name__ == "__main__":
    _test_class()
