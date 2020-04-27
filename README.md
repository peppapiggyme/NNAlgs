# Introduction
- package for neural network algorithm training and tuning
- `uproot`+`keras`+`lmdb`
- bowen.zhang@cern.ch

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

# Installation
- (updated on 27/04/2020 using)
```shell script
conda env export > CondaEnv.yml
```
- then restore the environments
```shell script
conda env create -f CondaEnv.yml
```

# [TBD] How to add a new project xxx


# Current workspace
>["work-2020" on Jupyter-Lab (US), valid until 28/09/2020](https://work-2020.atlas-ml.org/)
- 6 GPU (GeForce RTX 2080 Ti)
- 16 CPU (Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz)
- 64GB RAM

# TODOs
- **Testing**
- ~~using MxAOD instead of flat ROOT tree~~
- increase GPU usage [1]

[1] [What does GPU-Util means in nvidia-smi](https://stackoverflow.com/questions/40937894/nvidia-smi-volatile-gpu-utilization-explanation/40938696)
