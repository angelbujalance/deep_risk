#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=ECG_FT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=output/fine_tune_ecg/medium3D_ALL/coronary_artery_disease.out

# Activate your environment
source activate cmr_pretrain

# Directory containing ECG samples

# DATA DIRECTORIES FOR GRID SEARCH - SMALL DATASET
TRAIN_DIR="/projects/prjs1252/CL_data/ECG_data/ECG_balanced_coronary_artery_disease_train_leads.pt"
VAL_DIR="/projects/prjs1252/CL_data/ECG_data/ECG_balanced_coronary_artery_disease_val_leads.pt"
TEST_DIR="/projects/prjs1252/CL_data/ECG_data/ECG_balanced_coronary_artery_disease_test_leads.pt"

TRAIN_LABELS_PTH="/projects/prjs1252/CL_data/ECG_data/ECG_balanced_coronary_artery_disease_train_labels.csv"
VAL_LABELS_PTH="/projects/prjs1252/CL_data/ECG_data/ECG_balanced_coronary_artery_disease_val_labels.csv"
TEST_LABELS_PTH="/projects/prjs1252/CL_data/ECG_data/ECG_balanced_coronary_artery_disease_test_labels.csv"

# OUTPUT DIR TO SAVE MODEL
OUT_DIR='/projects/prjs1252/CL_data/checkpoints/MAE_ECG_finetuned/CL'

checkpoint="$HOME/deep_risk/mae/MAE_pretrain/checkpoint-141-loss-0.1758.pth" # medium
# CL checkpoint
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium2/ECGEncoder_checkpoint-15-loss-3.334488876071977.pth" # CL medium
# checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D/ECGEncoder_checkpoint-9-loss-3.60972042289781.pth" # CL 3D data
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_all_labels_ld_258_lstm_768/ECGEncoder_checkpoint-6-loss-3.9472279578079412.pth"
# checkpoint="$HOME/deep_risk/mae/MAE_pretrain/checkpoint-141-loss-0.1758.pth" # medium
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_all_labels/ECGEncoder_checkpoint-12-loss-3.4776823461791615.pth" # CL 3D data ALL LABELS

#export CUDA_LAUNCH_BLOCKING=1

# 30 nans
# Grid search params
learning_rates=(1e-2 1e-3 1e-4 1e-5)
weight_decays=(1e-2 1e-3 1e-4)
accum_iter=(1 2)

# GRID SEARCH BEST PARAMS
learning_rates=(1e-4)
weight_decays=(1e-4)
accum_iter=(1)

endpoint_codes=("coronary_artery_disease" "atrial_fibrilation" "sudden_cardiac_death" "heart_failure" "myocardial_infarction" "cardiomyopathy")

# Grid search loop
for lr in "${learning_rates[@]}"
do
    for wd in "${weight_decays[@]}"
    do
        for ai in "${accum_iter[@]}"
        do
            echo ""
            echo ""
            echo "Running with lr=$lr, weight_decay=$wd, accum_iter=$ai"

            python fine_tune_ecg.py --model_name mae_vit_mediumDeep_patchX \
                --batch_size 64 \
                --train_path ${TRAIN_DIR} \
                --val_path ${VAL_DIR} \
                --test_path ${TEST_DIR} \
                --train_labels_path ${TRAIN_LABELS_PTH} \
                --val_labels_path ${VAL_LABELS_PTH} \
                --test_labels_path ${TEST_LABELS_PTH} \
                --num_outputs 1 \
                --epochs 1 \
                --accum_iter ${ai} \
                --checkpoint_path ${checkpoint} \
                --lr ${lr} \
                --weight_decay ${wd}
        done
    done
done
                 
