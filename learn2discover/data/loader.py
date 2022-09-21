import os
import pandas as pd
from data.schema import Schema, VarType
from configs.config_manager import ConfigManager
from loggers.logger_factory import LoggerFactory

class Loader:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__class__.__name__)
        self.config_manager = ConfigManager.get_instance()
        self.workspace_dir = self.config_manager.workspace_dir
        self.data_path = os.path.join(self.config_manager.workspace_dir, self.config_manager.data_file)

    def load_data(self) -> (Schema, pd.DataFrame):
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
        # Validity checks
        for i in range(len(data.columns)):
            cname = data.columns[i]
            # Check that the schema size corresponds to the number of variables
            assert len(schema.keys()) == len(data.columns)
            # Check that all values in the categorical data are represented in the schema
            if schema.get_type(cname) == VarType.CATEGORICAL:
                unique_categories = set(data.iloc[:,i].astype(str))
                schema_categories = set(schema.get_variable_values(cname))
                assert unique_categories.issubset(schema_categories), str(unique_categories) + str(schema_categories)

    def _read_csv_file(self, schema: Schema) -> pd.DataFrame: 
        if not os.path.exists(self.data_path):
            raise FileNotFoundError
        with open(self.data_path, 'r') as d:
            index_column_included = False if not self.config_manager.index_column_included else 0
            data = pd.read_csv(d, sep=self.config_manager.delimiter, 
                                index_col=index_column_included, 
                                header=0, names=schema.keys())  # always use names from schema
        return data