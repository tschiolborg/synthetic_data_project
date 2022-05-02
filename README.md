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



## Project Structure


```
├── conf                    <- Hydra configuration files
│
├── data                    <- Project data
│
├── outputs                 <- Logs generated by Hydra and loggers
│
├── notebooks               <- Jupyter notebooks
│
├── reports                 <- Reports, results, notes, pdfs, figures etc.
│
├── src                     <- Source code
│   │
│   ├── config.py               <- @dataclasses describing the config files
│   ├── datasets.py             <- PyTorch Datasets
│   ├── engine.py               <- Training and test scripts
│   ├── models.py               <- Classifiers
│   ├── transforms.py           <- Transformations
│   └── utils.py                <- Utility scripts
│
├── train.py                <- Run training
├── test.py                 <- Run testing
│
├── .env                    <- Private environment variables
├── .gitignore              <- List of files/folders ignored by git
├── requirements.txt        <- File for installing python dependencies
├── setup.cfg               <- Configuration of linters
├── pyproject.toml          <- Configuration of black
└── README.md
```


--------



