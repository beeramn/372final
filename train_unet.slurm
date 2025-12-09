#!/bin/bash
#SBATCH -p gpu-common,scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=train
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/bin/activate unet_env

cd 372final
python train_unet.py