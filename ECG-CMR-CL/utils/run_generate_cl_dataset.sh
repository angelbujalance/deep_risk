#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --job-name=SEG_CMR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SMATCH --mem=160GB
#SBATCH --output=output/generate_balanced_dataset2.out

ECG_DIR="$HOME/deep_risk/data"
CMR_DIR='/projects/prjs1252/CL_data'
OUT_DIR='/projects/prjs1252/CL_data/ECG_data'


# python generate_cl_dataset.py --ecg_data_dir ${ECG_DIR} \
#                                --cmr_data_dir ${CMR_DIR} \
#                                --output_dir ${OUT_DIR}

# python generate_finetune_dataset.py --output_dir ${OUT_DIR}

# python generate_balanced_dataset.py

#python generate_clinical_data.py --ecg_data_dir ${ECG_DIR} \
#                                  --output_dir ${OUT_DIR}

# echo "Generate clinical Data finished succesfully!"

python generate_balanced_dataset.py
