Synthetic data project
==============================

Synthetic data for traing deep neural networks for object detection on traffic signs.

Read more under [reports](reports).


## Requirements
Virtual environment and dependencies with `conda`:
```bash
conda create --name [name] python=3.9
```

Install requirements (in virtual environment): 
```bash
pip install -r requirements.txt
```



# Project Structure


```
├── conf                    <- Hydra configuration files
│   ├── augmentations           <- Augmentations / transformations configs
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── debug                   <- Debugging configs
│   ├── experiment              <- Experiment configs
│   ├── hparams_search          <- Hyperparameter search configs
│   ├── log_dir                 <- Logging directory configs
│   ├── logger                  <- Logger configs
│   ├── model                   <- Model configs
│   ├── trainer                 <- Trainer configs
│   ├── training                <- Training configs
│   │
│   ├── debug.yaml            <- Main config for debugging. Used for unit testing.
│   ├── test.yaml             <- Main config for testing
│   └── train.yaml            <- Main config for training
│
├── data                    <- Project data
│
├── logs                    <- Logs generated by Hydra and PyTorch Lightning loggers
│
├── notebooks               <- Jupyter notebooks
│
├── reports                 <- Reports, results, notes, pdfs, figures etc.
│
├── src                     <- Source code
│   ├── datamodules             <- Lightning datamodules
│   ├── datasets                <- PyTorch Dataset classes
│   ├── models                  <- Lightning models
│   ├── utils                   <- Utility scripts
│   │
│   ├── config.py               <- dataclasses describing the config files
│   └── train.py
│
├── tests                   <- Tests of any kind
│   ├── visual                   <- Test visual by looking at images
│   └── unit                     <- Unit tests
│
├── train.py              <- Run training
│
├── .env                      <- Private environment variables
├── .gitignore                <- List of files/folders ignored by git
├── requirements.txt          <- File for installing python dependencies
├── setup.cfg                 <- Configuration of linters and pytest
├── black.toml                <- Configuration of black
└── README.md
```


--------



