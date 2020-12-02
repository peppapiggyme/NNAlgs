# Introduction
- package for neural network algorithm training and tuning
- `uproot`+`lmdb`+`keras/keras-tuner`
- bowen.zhang@cern.ch

# Installation

- install anaconda

- then restore the environments
```shell script
conda env create -f CondaEnv.yml
```

>- (updated on 04/06/2020 using)
>```shell script
>conda env export > CondaEnv.yml
>```

- git clone this repository

# How to run it

```shell script
# to train with static architecture
python NNAlgs Train NNAlgs/config/ConfDSNN.yaml
# to tune hyper-parameters
python NNAlgs Tune NNAlgs/config/ConfDSNN.yaml
```

# How to add a new project

# Layout
```text
.
├── CondaEnv.yml
├── README.md
├── __main__.py
├── config
│   ├── ArchDSNN.yaml
│   ├── ArchLSTM.yaml
│   ├── ConfDSNN.yaml
│   ├── ConfLSTM.yaml
│   ├── DatasetBuilder.py
│   ├── GeneratorLoaders.py
│   └── __init__.py
├── data
│   ├── json
│   └── lmdb
├── nnalgs
│   ├── __init__.py
│   ├── algs
│   │   ├── DataGenerators.py
│   │   ├── LMDBCreators.py
│   │   ├── Models.py
│   │   └── __init__.py
│   ├── base
│   │   ├── BuilderDirector.py
│   │   ├── DataGenerator.py
│   │   ├── DatasetBuilder.py
│   │   ├── GeneratorLoader.py
│   │   ├── IDataset.py
│   │   ├── LMDBCreator.py
│   │   └── __init__.py
│   └── utils
│       ├── CallBack.py
│       ├── Common.py
│       ├── Enum.py
│       ├── LMDBGen.py
│       ├── Logger.py
│       ├── Math.py
│       └── __init__.py
└── nnexec
    ├── TrainDecayMode.py
    ├── TuneDecayMode.py
    └── __init__.py
```

# TODOs
- increase GPU usage [1]

[1] [What does GPU-Util means in nvidia-smi](https://stackoverflow.com/questions/40937894/nvidia-smi-volatile-gpu-utilization-explanation/40938696)
