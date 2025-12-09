#!/bin/bash
#SBATCH -p gpu-common 
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --job-name=train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/bin/activate unet_env

cd 372final
python train_unet.py