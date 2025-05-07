#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --job-name=SEG_CMR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SMATCH --mem=160GB
#SBATCH --output=output/create_cl_dataset.out

ECG_DIR="$HOME/deep_risk/data"
CMR_DIR='/projects/prjs1252/CL_data'
OUT_DIR='/projects/prjs1252/CL_data/CL_3D'

python generate_cl_dataset.py --ecg_data_dir ${ECG_DIR} \
                               --cmr_data_dir ${CMR_DIR} \
                               --output_dir ${OUT_DIR}

# python generate_finetune_dataset.py --output_dir ${OUT_DIR}

# python generate_balanced_dataset.py --output_dir ${OUT_DIR}