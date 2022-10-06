call `python l2d.py` in `learn2discover/`

# Project Layout
        ├── data_generation   <-- code required to generate labelled datasets
        ├── datasets
        │   ├── generated/    <-- generated datasets, to be used as input to model training
        │   ├── original/     <-- original datasets
        │   └── schemas/      <-- schemas for differentiating inputs vs outputs, and var types
        ├── learn2discover
        │   ├── config.yml    <-- default config file
        │   ├── configs/      <-- contains loader class for main config file
        │   ├── data/         <-- helper classes for managing data
        │   ├── l2d.py        <-- main executable
        │   ├── loggers/
        │   ├── oracle/
        │   └── utils/        <-- additional helper classes
        ├── LICENSE
        ├── README.md
        └── requirements.txt  <-- library versions used in development