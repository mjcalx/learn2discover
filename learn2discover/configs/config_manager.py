import os
import yaml


class ConfigManager:
    config_file = 'config.yml'
    instance = None
    data_filepath = ''

    primary_logger_type = 'console'
    log_level = 'info'
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir

    def get_yaml_configs(self):
        yaml_path = os.path.join(self.workspace_dir, self.config_file)
        print(yaml_path)
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
            print('Error!')

    @staticmethod
    def create_instance(workspace_dir):
        if ConfigManager.instance is None:
            ConfigManager.instance = ConfigManager(workspace_dir)

        return ConfigManager.instance

    def load_configs(self):
        configs = self.get_yaml_configs()
        print('configs',configs)
        self.data_filepath = configs.get('sut_settings').get('data_filepath')
        self.primary_logger_type = configs.get('log').get('primary_logger_type')
        self.log_level = configs.get('log').get('log_level')
