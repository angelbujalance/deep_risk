#!/bin/bash

#SBATCH --partition=fat_rome
#SBATCH --job-name=CL_DAT_LABELS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=output/generate_balanced_LVEF_dataset_2.out

ECG_DIR="/projects/prjs1252/CL_data/ECG_data"
CMR_DIR='/scratch-shared/abujalancegome/CL_data'
OUT_DIR='/projects/prjs1252/CL_data/ECG_data/normalized'

ECG_DIR='/projects/prjs1252/CL_data/ECG_data/'



# python generate_cl_dataset.py --ecg_data_dir ${ECG_DIR} \
#                                 --cmr_data_dir ${CMR_DIR} \
#                                 --output_dir ${OUT_DIR}

# python generate_finetune_dataset.py --ecg_data_dir ${ECG_DIR} \
#                                     --output_dir ${OUT_DIR}

python generate_balanced_dataset.py --ecg_data_dir ${ECG_DIR} \
                                    --output_dir ${OUT_DIR}

#python generate_clinical_data.py --ecg_data_dir ${ECG_DIR} \
#                                  --output_dir ${OUT_DIR}

# echo "Generate clinical Data finished succesfully!"

# python generate_CL_balanced_dataset.py
