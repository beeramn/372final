## Project Setup Guide

This document describes how to install dependencies and perform audio source separation using our model. 

## Cloning

To clone this repository, run the command: 

`git clone https://github.com/beeramn/372final.git`

## Environment Setup

## Option A: Download Dependencies

To download the dependencies using `pip`, run the following command: 

`pip install -r requirements.txt`

Alternatively, you can run: 

`pip install torch torchaudio soundfile numpy librosa tqdm matplotlib`

## Option B: Create Conda Environment

To setup a conda environment, run the following commands: 

`conda env create -f environment.yml`
`conda activate unet_env`

## Performing Source Separation

To perform source separation, simply run the following in the command line: 

`python inference.py path/to/song.wav`

For example, you can test your installation by running inference.py on WASTE, the testing track we have provided on the root level of the repository: 

`python inference.py WASTE_BROCKHAMPTON.wav`
