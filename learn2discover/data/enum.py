from enum import Enum

class Label(Enum):
    """
    A binary label with values "Fair" and "Unfair"
    """
    FAIR = 'FAIR'
    UNFAIR = 'UNFAIR'


class Outcome(Enum):
    """
    A binary outcome with values "Pass" and "Fail"
    """
    PASS = 'PASS'
    FAIL = 'FAIL'


class ParamType(Enum):
    INPUTS = "INPUTS"
    OUTPUTS = "OUTPUTS"
    OUTCOME = "OUTCOME"
    FAIRNESS = "FAIRNESS"


class VarType(Enum):
    CATEGORICAL = 'categorical'
    NUMERICAL = 'numerical'
    DATE = 'date'
    ID = 'id'
