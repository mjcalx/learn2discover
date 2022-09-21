import os
import yaml

class LoaderUtils:

    @staticmethod
    def get_yaml_configs(workspace_dir=os.getcwd(), config_file=None):
        yaml_path = os.path.join(workspace_dir, config_file)
        configs = {}
        if os.path.exists(yaml_path) and config_file is not None:
            stream = open(yaml_path, "r")
            configs = yaml.full_load(stream)
            return configs
        raise FileNotFoundError