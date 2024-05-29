# NeuroVision - A tool for brain tumor classification and segmentation

This repository allows to perform brain tumor classification and segmentation based on MRI images. Furthermore, this repository provides a streamlit web application called "NeuroVision - A tool for brain tumor classification and segmentation" to perform brain tumor classification and segmentation online. 

To use this repository follow the following steps:
* 1) Clone repository 
* 2) Install the virtual environment
* 3) Download datasets (source: kaggle):
    * 3.1) Dataset: Brain tumor classification (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
    * 3.2) Dataset: Brain tumor segmentation (https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset/data)
* 4) Run notebooks included in repository
* 5) Run streamlit.py to launch streamlit web application

## Requirements:

- pyenv with Python: 3.11.3

## Setup virtual environment: 

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

The `requirements.txt` file contains the libraries needed for running the notebooks included in this repository.