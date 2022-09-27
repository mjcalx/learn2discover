import os
import yaml
from utils.loader_utils import LoaderUtils


class ConfigManager:
    MODE_STR_TRAINING = 'train'
    MODE_STR_GENERATE = 'generate'
    
    CONFIG_FILE_TRAINING = 'config.yml'
    CONFIG_FILE_GENERATE = '../data_generation/config.yml'
    MODES = {
        MODE_STR_TRAINING : CONFIG_FILE_TRAINING, 
        MODE_STR_GENERATE : CONFIG_FILE_GENERATE
        }

    config_file = CONFIG_FILE_TRAINING
    schema_file = ''
    data_file = ''
    column_names_included = False
    index_column_included = False
    instance = None
    model_path = ''
    training_path = ''
    validation_path = ''
    evaluation_path = ''
    unlabelled_path = ''
    fair_csv_filename = 'fair.csv'
    unfair_csv_filename = 'unfair.csv'

    primary_logger_type = 'console'
    log_level = 'info'


    def __init__(self, 
                 workspace_dir : str, 
                 mode : str=None,
                 config_file : str=None):
        if mode not in ConfigManager.MODES.keys():
            print(f'Defaulting to "{ConfigManager.MODE_STR_TRAINING}" mode')
            mode = ConfigManager.MODE_STR_TRAINING
        self.mode = mode
        self.config_file = config_file if config_file is not None else ConfigManager.MODES[self.mode]
        if ConfigManager.instance is None:
            ConfigManager.instance = self
        self.workspace_dir = workspace_dir
        self.load_configs()

    def get_yaml_configs(self, config_file=None):
        cfg = config_file if config_file is not None else self.config_file
        return LoaderUtils.get_yaml_configs(self.workspace_dir, cfg)

    @staticmethod
    def get_instance():
        if ConfigManager.instance is not None:
            return ConfigManager.instance
        else:
            print('ERROR: Can\'t find ConfigManager instance. Please '\
                  'instantiate a ConfigManager + configs before calling get_instance()')

    def get_data_schema(self, schema_file=None):
        schf = schema_file if schema_file is not None else self.schema_file
        return self.get_yaml_configs(schf)

    def load_configs(self):
        self.configs = self.get_yaml_configs()
        load_fn = {
            ConfigManager.MODE_STR_TRAINING : self._load_training_configs,
            ConfigManager.MODE_STR_GENERATE : self._load_data_gen_configs
        }[self.mode]
        load_fn()
        self.primary_logger_type = self.configs.get('log').get('primary_logger_type')
        self.log_level = self.configs.get('log').get('log_level')

    def _load_data_gen_configs(self) -> None:
        self.schema_file           = self.configs.get('generator_settings').get('schema_file')
        self.data_file             = self.configs.get('generator_settings').get('data_file')
        self.index_column_included = self.configs.get('generator_settings').get('index_column_included')
        self.delimiter             = self.configs.get('generator_settings').get('delimiter')
        
        self.output_csv = self.configs.get('output_settings').get('output_csv')

        assert isinstance(self.index_column_included, bool)

    def _load_training_configs(self) -> None:
        self.schema_file           = self.configs.get('dataset_settings').get('schema_file')
        self.data_file             = self.configs.get('dataset_settings').get('data_file')
        self.index_column_included = self.configs.get('dataset_settings').get('index_column_included')
        self.delimiter             = self.configs.get('dataset_settings').get('delimiter')

        self.model_path          = self.configs.get('output_settings').get('model_path')
        self.training_path       = self.configs.get('output_settings').get('training_path')
        self.validation_path     = self.configs.get('output_settings').get('validation_path')
        self.evaluation_path     = self.configs.get('output_settings').get('evaluation_path')
        self.unlabelled_path     = self.configs.get('output_settings').get('unlabelled_path')
        self.fair_csv_filename   = self.configs.get('output_settings').get('fair_csv_filename')
        self.unfair_csv_filename = self.configs.get('output_settings').get('unfair_csv_filename')

        self.query_strategies  = self.configs.get('training_settings').get('query_strategies')

        assert isinstance(self.column_names_included, bool)
        assert isinstance(self.index_column_included, bool)
        assert isinstance(self.query_strategies, list)