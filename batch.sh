#!/bin/sh
#SBATCH -p gpi.compute
#SBATCH -c 2
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1,gpumem:6G
python xgoost_train_models.py
