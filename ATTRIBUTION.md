## Overview

This document lists all external resources, libraries, datasets, code snippets, and tools used in the development of this project. 

## Libraries & Frameworks

The following open-source libraries were used in accordance with their respective licenses:

PyTorch
https://pytorch.org/ 
Used for building and training the U-Net model (tensors, autograd, DataLoader, optimizers).

Torchaudio
https://pytorch.org/audio/stable/ 
Used for audio loading and resampling.

NumPy
https://numpy.org/ 
Used for general numerical operations.

SoundFile (libsndfile)
https://python-soundfile.readthedocs.io/en/0.13.1/ 
Used for reading and writing WAV files.

SciPy (STFT reference)
https://scipy.org/ 
Documentation consulted for understanding STFT/ISTFT relationships.

## Dataset Sources

MUSDB18 / MUSDB 2-stem derivative
https://sigsep.github.io/datasets/musdb.html 
Used as the source of mixture, vocals, and instrumental stems.

Our final training dataset was created by combining training data from MUSDB18 (which we modified to synthesize the separate instrumental tracks) with the following 13 tracks which we personally curated and scraped for training/testing purposes. 

BROCKHAMPTON - WASTE
The Strokes - Someday
Juice WRLD - Lucid Dreams
Laufey - From the Start
Sabrina Carpenter - Espresso
LE SSERAFIM - Blue Flame
Don Toliver - Private Landing
Frank Ocean - Pink & White
keshi - Soft Spot
Post Malone - Circles
Peach Pit - Black Licorice
PinkPantheress - Mosquito
Clairo - Sofia

The 2-stem data for the latter 12 tracks can be found in the following link: 

https://drive.google.com/drive/folders/1raUwtb8_I8avzisCFgRwgfngshMM4qWz?usp=drive_link 

## Tools & Environment

Training was performed on the Duke compute cluster using SLURM jobs. 

## AI Assistance Disclosure

Portions of this project were developed with the assistance of ChatGPT (OpenAI). 
AI assistance was used in the following ways:
Debugging support: Identifying causes of runtime errors, CUDA memory issues, and mismatched tensor shapes.
Code refinements: Suggestions for restructuring U-Net modules, reducing memory usage, and improving inference stability.
Template generation: Drafting skeletons for training loops and inference pipelines. 

AI-generated suggestions were manually reviewed and adapted.

## Acknowledgments

Special thanks to Prof. Fain and the CS372 course staff for their guidance on deep learning best practices. 
