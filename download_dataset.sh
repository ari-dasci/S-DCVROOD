#!/bin/bash 

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate novelty_detection
export PYTHONPATH='.'

python src/download_datasets.py
