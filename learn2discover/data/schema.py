from loggers.logger_factory import LoggerFactory
from collections import OrderedDict

class Schema(OrderedDict):
    DESCRIPTION_STR = 'description'
    TYPE_STR        = 'type'
    CATEGORICAL_STR = 'categorical'
    NUMERICAL_STR   = 'numerical'
    VALUES_STR      = 'values'
    ERR_MISSING_VALUES_KEY    = 'MissingValuesKey'
    ERR_NON_UNIQUE_VALUES     = 'NonUniqueValues'
    ERR_EMPTY_VALUES_DICT     = 'EmptyValuesDict'
    ERR_BAD_VARIABLE_TYPE     = 'BadVariableType'
    ERR_MULTIPLE_OR_NO_LABELS = 'MultipleOrNoLabels'

    def __init__(self, parsed_yaml: dict):
        super(Schema, self).__init__(parsed_yaml)
        self.logger = LoggerFactory.get_logger(__class__.__name__) 
        self._validate()
        
    
    def get_type(self, key: str) -> str:
        try:
            return self[key][self.TYPE_STR]
        except KeyError:
            self.logger.error(f'KeyError: "{key}" for schema when retrieving type')
            raise

    def get_variable_values(self, key: str) -> tuple:
        """
        Values are strings if categorical; otherwise returns an empty tuple
        """
        try:
            if self.get_type(key) == self.CATEGORICAL_STR:
                return tuple(self[key][self.VALUES_STR])
            if self.get_type(key) == self.NUMERICAL_STR:
                return ()
            raise ValueError
        except KeyError:
            self.logger.error(f'KeyError: "{key}" for schema when retrieving variable values')
            raise

    def _validate(self):
        """Check that data meets constraints/is valid"""

        try:
            # Check that :
            #   - categorical values are defined
            #   - categorical values are unique
            for k in self.keys():
                var_type = self.get_type(k)
                if var_type == self.CATEGORICAL_STR:
                    if self.VALUES_STR not in self[k].keys():
                        raise KeyError(self.ERR_MISSING_VALUES_KEY)
                    values_dict = self[k][self.VALUES_STR]
                    if not isinstance(values_dict, dict) or not len(values_dict) > 0:
                        raise KeyError(self.ERR_EMPTY_VALUES_DICT)
                    if len(set(values_dict.values())) != len(values_dict):
                        raise ValueError(self.ERR_NON_UNIQUE_VALUES)
                elif var_type == self.NUMERICAL_STR:
                    pass
                else:
                    raise ValueError(self.ERR_BAD_VARIABLE_TYPE)
        except ValueError as e:
            msg = ''
            if str(e) == self.ERR_BAD_VARIABLE_TYPE:
                msg = ': type of variable must be "{0}" or "{1}". Received "{2}"'.format(
                        self.NUMERICAL_STR, self.CATEGORICAL_STR, var_type)
            if str(e) == self.ERR_NON_UNIQUE_VALUES:
                msg = ': data values must be unique'
            self.logger.error(f'{e} {msg}')
            raise
        except KeyError as e:
            self.logger.error(f'{e} : categorical variables must have non-empty dict of unique values')
            raise
            raise


    def get_description(self, variable_name: str) -> str:
        """Return description, if given in the schema. Else return the variable name."""
        try:
            if self.DESCRIPTION_STR in self[variable_name].keys():
                return self[variable_name][self.DESCRIPTION_STR]
            return variable_name
        except KeyError:
            self.logger.error(f"KeyError: {variable_name}")
