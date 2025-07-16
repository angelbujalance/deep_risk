#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=CL_3Dvol
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --output=output/cl_train_ViT_medium3D_diastole_testing_CL_w_labels.out

# Activate your environment
source activate cmr_pretrain

# Directory containing ECG samples

# DATA DIRECTORIES FOR GRID SEARCH - SMALL DATASET
# CMR_DATA_DIR='/scratch-shared/abujalancegome/CMR_data/cmr_batch_0.pt' # 50 CMR samples
# ECG_DATA_DIR='/home/abujalancegome/deep_risk/data/tensor_toy.pt'
ECG_TRAIN_DIR='/projects/prjs1252/CL_data/CL_ECG_leads_train.pt'
ECG_VAL_DIR='/projects/prjs1252/CL_data/CL_ECG_leads_val.pt'
CMR_TRAIN_DIR='/projects/prjs1252/CL_data/CL_cmr_tensor_train.pt'
CMR_VAL_DIR='/projects/prjs1252/CL_data/CL_cmr_tensor_val.pt'

export CUDA_LAUNCH_BLOCKING=1

# Grid search params
learning_rates=(1e-3 1e-4 1e-5 1e-6)
weight_decays=(1e-3 1e-4 1e-5)
# mask_ratios=(0.7)

# learning_rates=(1e-3)
# weight_decays=(1e-3)
accum_iter=(1 2)

learning_rates=(1e-4)
weight_decays=(1e-5)
# mask_ratios=(0.7)

# learning_rates=(1e-3)
# weight_decays=(1e-3)
accum_iter=(1)

# CHECKPOINT CMR (WITH 3D VOLUME)
checkpoint_path_cmr="/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/3D_diastole/ResNet503D_MLP_checkpoint-99-loss-107.25341754489475-lr-0.0001-wd-1e-5-ai-1.pth"

ECG_TRAIN_DIR='/projects/prjs1252/CL_data/CL_3D/CL_ECG_leads_balanced_atrial_fibrilation_diastole_train_3d.pt'
ECG_VAL_DIR='/projects/prjs1252/CL_data/CL_3D/CL_ECG_leads_balanced_atrial_fibrilation_diastole_val_3d.pt'

CMR_TRAIN_DIR='/projects/prjs1252/CL_data/CL_3D/CL_cmr_tensor_balanced_atrial_fibrilation_diastole_train_3d.pt'
CMR_VAL_DIR='/projects/prjs1252/CL_data/CL_3D/CL_cmr_tensor_balanced_atrial_fibrilation_diastole_val_3d.pt'

TRAIN_LABELS="/projects/prjs1252/CL_data/CL_3D/ECG_balanced_atrial_fibrilation_diastole_train_labels.csv"
VAL_LABELS="/projects/prjs1252/CL_data/CL_3D/ECG_balanced_atrial_fibrilation_diastole_val_labels.csv"

# CHECKPOINT ECG
# mae_vit_mediumDeep_patchX, vit_base_patch200
checkpoint_path_ecg='/home/abujalancegome/deep_risk/mae/MAE_pretrain_chkpts_/checkpoint-94-loss-0.1754.pth' # base
checkpoint_path_ecg="$HOME/deep_risk/mae/MAE_pretrain/checkpoint-141-loss-0.1758.pth" # medium

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

            python train_cl.py --model_name ResNet50-3D-MLP \
                --ecg_model mae_vit_mediumDeep_patchX \
                --batch_size 64 \
                --cmr_train_path ${CMR_TRAIN_DIR} \
                --ecg_train_path ${ECG_TRAIN_DIR} \
                --cmr_val_path ${CMR_VAL_DIR} \
                --ecg_val_path ${ECG_VAL_DIR} \
                --checkpoint_path_cmr ${checkpoint_path_cmr} \
                --checkpoint_path_ecg ${checkpoint_path_ecg} \
                --train_labels_path ${TRAIN_LABELS} \
                --val_labels_path ${VAL_LABELS} \
                --num_outputs 3 \
                --temporal_dim 6 \
                --output_dir CL_results/medium3D_diastole_w_labels_atrial \
                --epochs 200 \
                --latent_dim 768 \
                --projection_dim 128 \
                --accum_iter ${ai} \
                --lr ${lr} \
                --weight_decay ${wd}
        done
    done
done