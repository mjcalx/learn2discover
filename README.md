call `python l2d.py` in `learn2discover/`

For data generation: 

1. modify the `config.yml` in the `data_generation` directory
1. subclass `SystemUnderTest` and `AbstractMockOracle`
1. modify `python generate_data.py` to use your new subclasses
1. call `python generate_data.py`

# Project Layout
        ├── data_generation            <-- code required to generate labelled datasets
        ├── datasets
        │   ├── generated/             <-- generated datasets, to be used as input to model training
        │   ├── original/              <-- original datasets
        │   └── schemas/               <-- schemas for differentiating inputs vs outputs, and var types
        ├── learn2discover
        │   ├── config.yml             <-- default config file
        │   ├── configs/               <-- contains loader class for main config file
        │   ├── data/                  <-- helper classes for managing data
        │   ├── l2d.py                 <-- main executable
        │   ├── loggers/      
        │   ├── oracle/
        │   │   ├── query_strategies/
        │   │   └── stopping_criteria/
        │   └── utils/                 <-- additional helper classes
        ├── LICENSE
        ├── README.md
        └── requirements.txt  <-- library versions used in development

