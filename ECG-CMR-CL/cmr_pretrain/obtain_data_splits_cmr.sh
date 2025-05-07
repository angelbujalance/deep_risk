#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=tr_va_te
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=output/train_val_test_splits.out


DATA_DIR='/scratch-shared/abujalancegome/CMR_data/cmr_tensors_ts_all.pt'
OUT_DIR='/projects/prjs1252/CL_data'

python train_val_test_cmr.py --data_path ${DATA_DIR} --output_dir ${OUT_DIR}