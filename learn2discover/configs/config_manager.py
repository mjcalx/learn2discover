from utils.loader_utils import LoaderUtils


class ConfigManager:
    MODE_STR_TRAINING = 'train'
    MODE_STR_GENERATE = 'generate'

    CONFIG_FILE_TRAINING = 'config.yml'
    CONFIG_FILE_GENERATE = '../data_generation/config.yml'
    MODES = {
        MODE_STR_TRAINING: CONFIG_FILE_TRAINING,
        MODE_STR_GENERATE: CONFIG_FILE_GENERATE
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

    input_attrs = []
    output_attrs = []

    primary_logger_type = 'console'
    log_level = 'info'


    def __init__(self, 
                 workspace_dir: str,
                 mode: str=None,
                 config_file: str=None):
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
        self.verbosity = self.configs.get('log').get('verbosity')

    def _load_data_gen_configs(self) -> None:
        self.schema_file           = self.configs.get('generator_settings').get('schema_file')
        self.data_file             = self.configs.get('generator_settings').get('data_file')
        self.index_column_included = self.configs.get('generator_settings').get('index_column_included')
        self.delimiter             = self.configs.get('generator_settings').get('delimiter')
        self.input_attrs           = self.configs.get('generator_settings').get('input_attrs')
        self.output_attrs          = self.configs.get('generator_settings').get('output_attrs')
        self.sut_path              = self.configs.get('generator_settings').get('sut_path')
        self.sut_name              = self.configs.get('generator_settings').get('sut_name')
        self.oracle_path           = self.configs.get('generator_settings').get('oracle_path')
        self.oracle_name           = self.configs.get('generator_settings').get('oracle_name')
        
        self.output_csv = self.configs.get('output_settings').get('output_csv')

        assert isinstance(self.index_column_included, bool)

    def _load_training_configs(self) -> None:
        _section   = lambda section : self.configs.get(section)
        DATASET    = _section('dataset_settings')
        HYPERPARAM = _section('model_hyperparameters')
        TRAINING   = _section('training_settings')
        CRITERION  = TRAINING.get('stopping_criterion')
        OUTPUT     = _section('output_settings')

        self.dataset_settings      = DATASET
        self.schema_file           = DATASET.get('schema_file')
        self.data_file             = DATASET.get('data_file')
        self.index_column_included = DATASET.get('index_column_included')
        self.delimiter             = DATASET.get('delimiter')

        self.hyperparameter_settings = HYPERPARAM
        self.epochs                  = HYPERPARAM.get('epochs')
        self.learning_rate           = HYPERPARAM.get('learning_rate')
        self.selections_per_epoch    = HYPERPARAM.get('selections_per_epoch')
        self.dropout_rate            = HYPERPARAM.get('dropout_rate')
        self.layers                  = HYPERPARAM.get('layers')

        self.training_settings        = TRAINING
        self.test_fraction            = TRAINING.get('test_fraction')
        self.validation_fraction      = TRAINING.get('validation_fraction')
        self.unlabelled_sampling_size = TRAINING.get('unlabelled_sampling_size')
        self.query_strategy           = TRAINING.get('query_strategy')
        self.has_human_in_the_loop    = TRAINING.get('has_human_in_the_loop')

        self.stopping_criterion          = CRITERION.get('choice')
        self.stopping_criterion_settings = CRITERION.get('settings')[self.stopping_criterion]

        self.model_path          = OUTPUT.get('model_path')
        self.report_path         = OUTPUT.get('report_path')

        assert isinstance(self.column_names_included, bool)
        assert isinstance(self.index_column_included, bool)