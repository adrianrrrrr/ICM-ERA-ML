#!/bin/sh
#SBATCH -p gpi.compute             # Partition to submit to
#SBATCH --mem=64G      # Max CPU Memory
#SBATCH --gres=gpu:2
python xgoost_train_models.py
