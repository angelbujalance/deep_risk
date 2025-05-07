#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=data_splits
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=output/obtain_train_val_test_splits.out

python ecg_data_to_torch.py
python train_test_splits.py