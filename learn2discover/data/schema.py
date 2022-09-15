from loggers.logger_factory import LoggerFactory
from collections import OrderedDict

class Schema(OrderedDict):
    DESCRIPTION_STR = 'description'
    TYPE_STR        = 'type'
    CATEGORICAL_STR = 'categorical'
    NUMERICAL_STR   = 'numerical'
    VALUES_STR      = 'values'

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
        """Check that categorical values are defined"""
        try:
            for k in self.keys():
                var = self[k]
                var_type = var[self.TYPE_STR]
                if var_type == self.CATEGORICAL_STR:
                    assert self.VALUES_STR in self[k].keys(), 'MissingValuesKey'
                    assert isinstance(self[k][self.VALUES_STR], dict) and \
                            len(self[k][self.VALUES_STR]) > 0, 'EmptyValuesDict'
                elif var_type == self.NUMERICAL_STR:
                    pass
                else:
                    raise ValueError
            return
        except ValueError:
            self.logger.error(
                f'ValueError: Type of variable must be "{self.NUMERICAL_STR}"'
                f' or "{self.CATEGORICAL_STR}". Received "{var_type}"')
            raise
        except AssertionError as a:
            self.logger.error(f"AssertionError ({a}): categorical variables must "
                               "have non-empty list of valid values")
            raise

    def get_description(self, variable_name: str):
        """Return description, if given in the schema. Else return the variable name."""
        try:
            if self.DESCRIPTION_STR in self[variable_name].keys():
                return self[variable_name][self.DESCRIPTION_STR]
            return variable_name
        except KeyError:
            self.logger.error(f"KeyError: {variable_name}")
