from loggers.logger_factory import LoggerFactory

class Schema:
    DESCRIPTION_STR = 'description'
    TYPE_STR        = 'type'
    CATEGORICAL_STR = 'categorical'
    NUMERICAL_STR   = 'numerical'
    VALUES_STR      = 'values'

    def __init__(self, parsed_yaml: dict):
        self.schema = parsed_yaml
        self.logger = LoggerFactory.get_logger(__class__.__name__) 
        self._validate()

    def __getitem__(self, key):
        return self.schema[key]

    def keys(self):
        return self.schema.keys()
    
    def _validate(self):
        """Check that categorical values are defined"""
        try:
            for k in self.schema.keys():
                var = self.schema[k]
                var_type = var[self.TYPE_STR]
                if var_type == self.CATEGORICAL_STR:
                    assert self.VALUES_STR in self.schema[k].keys(), 'MissingValuesKey'
                    assert isinstance(self.schema[k][self.VALUES_STR], dict) and \
                            len(self.schema[k][self.VALUES_STR]) > 0, 'EmptyValuesDict'
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
            if self.DESCRIPTION_STR in self.schema[variable_name].keys():
                return self.schema[variable_name][self.DESCRIPTION_STR]
            return variable_name
        except KeyError:
            self.logger.error(f"KeyError: {variable_name}")
