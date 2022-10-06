import os
import pandas as pd
from data.schema import Schema, VarType
from data.data_classes import ParamType as param
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory

class Loader:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.config_manager = ConfigManager.get_instance()
        self.workspace_dir = self.config_manager.workspace_dir
        self.data_path = os.path.join(self.config_manager.workspace_dir, self.config_manager.data_file)
        self.is_training_mode = self.config_manager.mode == ConfigManager.MODE_STR_TRAINING

    def load_data(self) -> (Schema, pd.DataFrame):
        """
        Load YAML and CSV data into Schema and DataFrame objects respectively,
        and return these objects.

        We assume that the input CSV is multi-indexed; i.e., subframes can be 
        accessed with the keys defined in ParamType
        """
        try:
            # csv format: [ID, TEXT, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
            # desired csv format: [ID, COL1, ..., COLN, LABEL, SAMPLING_STRATEGY, CONFIDENCE]
            self.logger.debug("Loading schema...")
            cfgmgr = self.config_manager
            schema = Schema(cfgmgr.get_data_schema())
            # TODO
            # If training set, validation set, and eval set already exist...
            #     don't load the dataset, it's already been processed
            #     MAYBE perform per-entry check, for consistency.
            self.logger.debug("Loading dataset...")
            data = self._read_csv_file(schema)

            self._validity_checks(schema, data)

            return schema, data
        except FileNotFoundError as a:
            self.logger.error(f"FileNotFoundError: File {self.data_path} not found")
            raise

    def _validity_checks(self, schema: Schema, data: pd.DataFrame) -> bool:
        # Handle multi-indexed data in training mode
        #! Assumptions:
        #! - Data is never multiindexed in generation mode
        #! - Data is always multiindexed in training mode
        if isinstance(data.columns, pd.MultiIndex):
            data = pd.concat([data[param.INPUTS.value],data[param.OUTPUTS.value]], axis=1)
        # Check that the schema corresponds to the variables in the data
        set_columns = set(data.columns)
        assert set(schema.keys()) == set_columns
        # Check that all values in the categorical data are represented in the schema
        get_cname = lambda cname : cname if isinstance(cname, str) else cname[1]
        for i in range(len(data.columns)):
            cname = data.columns[i]
            if schema.get_type(cname) == VarType.CATEGORICAL:
                unique_categories = set(data.iloc[:,i].astype(str))
                schema_categories = set(schema.get_variable_values(cname))
                assert unique_categories.issubset(schema_categories), str(unique_categories) + str(schema_categories)

    def _read_csv_file(self, schema: Schema) -> pd.DataFrame: 
        if not os.path.exists(self.data_path):
            raise FileNotFoundError
        with open(self.data_path, 'r') as d:
            index_column_included = False if not self.config_manager.index_column_included else 0
            # Handle multi-indexed DataFrames if in training mode
            args = {'header' : 0, 'names' : schema.keys()} if not self.is_training_mode else {'header' : [0,1]}
            data = pd.read_csv(d, sep=self.config_manager.delimiter, 
                                index_col=index_column_included, 
                                **args) 
        return data