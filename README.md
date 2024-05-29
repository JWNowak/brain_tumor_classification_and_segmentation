# NeuroVision - A tool for brain tumor classification and segmentation

XXX

* XXX

## Requirements:

- pyenv with Python: 3.11.3

### Setup virtual environment: 

Use the requirements file in this repo to create a new virtual environment.

```BASH
make setup

#or

pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file contains the libraries needed for deployment.. of model or dashboard .. thus no jupyter or other libs used during development.