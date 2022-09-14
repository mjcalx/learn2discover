import os
import yaml


class ConfigManager:
    config_file = 'config.yml'
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

    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir

    def get_yaml_configs(self, config_file=None):
        cfg = config_file if config_file is not None else self.config_file
        yaml_path = os.path.join(self.workspace_dir, cfg)
        configs = {}
        if os.path.exists(yaml_path):
            stream = open(yaml_path, "r")
            configs = yaml.full_load(stream)
        return configs

    @staticmethod
    def get_instance():
        if ConfigManager.instance is not None:
            return ConfigManager.instance
        else:
            print('ERROR: Can\'t find ConfigManager instance')

    @staticmethod
    def create_instance(workspace_dir):
        if ConfigManager.instance is None:
            ConfigManager.instance = ConfigManager(workspace_dir)

        return ConfigManager.instance

    def get_data_schema(self):
        return self.get_yaml_configs(self.schema_file)

    def load_configs(self):
        configs = self.get_yaml_configs()

        self.schema_file           = configs.get('dataset_settings').get('schema_file')
        self.data_file             = configs.get('dataset_settings').get('data_file')
        self.column_names_included = configs.get('dataset_settings').get('column_names_included')
        self.index_column_included = configs.get('dataset_settings').get('index_column_included')
        self.delimiter             = configs.get('dataset_settings').get('delimiter')

        self.model_path          = configs.get('output_settings').get('model_path')
        self.training_path       = configs.get('output_settings').get('training_path')
        self.validation_path     = configs.get('output_settings').get('validation_path')
        self.evaluation_path     = configs.get('output_settings').get('evaluation_path')
        self.unlabelled_path     = configs.get('output_settings').get('unlabelled_path')
        self.fair_csv_filename   = configs.get('output_settings').get('fair_csv_filename')
        self.unfair_csv_filename = configs.get('output_settings').get('unfair_csv_filename')

        self.query_strategies  = configs.get('training_settings').get('query_strategies')
        self.primary_logger_type = configs.get('log').get('primary_logger_type')
        self.log_level = configs.get('log').get('log_level')

        assert isinstance(self.column_names_included, bool)
        assert isinstance(self.index_column_included, bool)
        assert isinstance(self.query_strategies, list)