#!/bin/sh
#SBATCH -p gpi.compute             # Partition to submit to
#SBATCH --mem=1G      # Max CPU Memory
#SBATCH --gres=gpu:1
python xgoost_train_models.py
