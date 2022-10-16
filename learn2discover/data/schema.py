from __future__ import annotations
from typing import List, Dict, Tuple
from data.enum import VarType
from loggers.logger_factory import LoggerFactory
from collections import OrderedDict

class Schema(OrderedDict):
    """
    This class is intended for future use with human-readable prompts for Human-in-the-Loop systems.
    """
    DESCRIPTION_STR = 'description'
    TYPE_STR        = 'type'
    VALUES_STR      = 'values'
    IS_LABEL_STR    = 'is_label'

    ERR_MISSING_VALUES_KEY    = 'MissingValuesKey'
    ERR_NON_UNIQUE_VALUES     = 'NonUniqueValues'
    ERR_EMPTY_VALUES          = 'EmptyValues'
    ERR_BAD_VARIABLE_TYPE     = 'BadVarType'
    ERR_MULTIPLE_OR_NO_LABELS = 'MultipleOrNoLabels'

    def __init__(self, parsed_yaml: Dict):
        super(Schema, self).__init__(parsed_yaml)
        self.logger = LoggerFactory.get_logger(__class__.__name__) 
        self.label_key = None
        self.types = {v:[] for v in VarType}
        self._validate()
    
    def get_type(self, key: str) -> str:
        try:
            return VarType(self[key][self.TYPE_STR])
        except KeyError:
            self.logger.error(f'KeyError: "{key}" for schema when retrieving type')
            raise
        except ValueError:
            return 'BadVarType'

    def get_variable_values(self, key: str) -> Tuple:
        """
        Values are strings if categorical; otherwise returns an empty tuple
        """
        try:
            if self.get_type(key) is VarType.CATEGORICAL:
                return tuple(map(lambda x : str(x), self[key][self.VALUES_STR]))
            if self.get_type(key) in VarType:
                return ()
            raise ValueError
        except KeyError:
            self.logger.error(f'KeyError: "{key}" for schema when retrieving variable values')
            raise

    def vars_by_type(self, vtype: VarType=None) -> Dict[VarType, List[str]]:
        assert vtype in [*VarType, None]
        if vtype is None:
            return self.types.copy()
        return self.types[vtype].copy()

    def get_label_key(self) -> str:
        assert self.label_key is not None
        return self.label_key

    def _validate(self) -> None:
        """Check that data meets constraints/is valid"""
        try:
            # Validity Checks
            for k in self.keys():
                var_type = self.get_type(k)
                # Check categorical values are defined and unique
                if var_type is VarType.CATEGORICAL:
                    self._validate_categories(k)
                # Check variable type is valid
                elif var_type not in VarType:
                    raise ValueError(self.ERR_BAD_VARIABLE_TYPE)
                self.types[var_type].append(k)
        except ValueError as e:
            msg = ''
            if str(e) == self.ERR_BAD_VARIABLE_TYPE:
                msg = ': type of variable must be in {0}. Received "{1}"'.format(
                        [i.value for i in list(VarType)], var_type)
            if str(e) == self.ERR_NON_UNIQUE_VALUES:
                msg = ': data values must be unique'
            self.logger.error(f'{e} {msg}')
            raise
        except KeyError as e:
            self.logger.error(f'{e} : categorical variables must have non-empty dict of unique values')
            raise

    def _validate_categories(self, key) -> None:
        if self.VALUES_STR not in self[key].keys():
            raise KeyError(self.ERR_MISSING_VALUES_KEY)
        values_read = self[key][self.VALUES_STR]
        if not (isinstance(values_read, dict) or isinstance(values_read, list)) or \
           not len(values_read) > 0:
            raise KeyError(self.ERR_EMPTY_VALUES)
        len_unique_values = (lambda x : len(set(x)) if isinstance(x, list) else len(set(x.values())))(values_read)
        if len_unique_values != len(values_read):
            raise ValueError(self.ERR_NON_UNIQUE_VALUES)
                

    def get_description(self, variable_name: str) -> str:
        """Return description, if given in the schema. Else return the variable name."""
        try:
            if self.DESCRIPTION_STR in self[variable_name].keys():
                return self[variable_name][self.DESCRIPTION_STR]
            return variable_name
        except KeyError:
            self.logger.error(f"KeyError: {variable_name}")
