#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=CL_med
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=output/cl_train_ViT_medium3D_all_labels_LSTM_freeze.out

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

# PATH TO PRE-TRAINED MODELS
checkpoint_path_cmr='/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/ResNet50_checkpoint-166-loss-393.5690521240234-lr-1e-06-wd-1e-5-ai-1.pth'
# checkpoint_path_cmr="$HOME/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/2D/ResNet50_checkpoint-194-loss-289.21529693603514-lr-0.0001-wd-1e-5-ai-1.pth"
checkpoint_path_cmr='/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/2Da/ResNet50_checkpoint-0-loss-18599.1482421875-lr-0.0001-wd-1e-5-ai-1.pth'
checkpoint_path_cmr='/home/abujalancegome/deep_risk/ResNet50_checkpoint-175-loss-392.58253479003906-lr-1e-06-wd-1e-5-ai-1.pth'
checkpoint_path_cmr='/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/2D/ResNet50_checkpoint-180-loss-365.94007568359376-lr-0.0001-wd-1e-5-ai-1.pth'

# CHECKPOINT CMR (WITH TIME) + 3D DATASETS
checkpoint_path_cmr='/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/3D/ResNet503D_checkpoint-151-loss-204.43801043110508-lr-0.0001-wd-1e-5-ai-1.pth'
# CHECKPOINT CMR (WITH TIME) + ALL LABELS
checkpoint_path_cmr='/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/3D_all_labelstest/ResNet503D_checkpoint-45-loss-81.18560766404674-lr-0.0001-wd-1e-5-ai-1.pth'
# checkpoint_path_cmr='/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/3D_all_labels_lstm_test_CL/ResNet503D_checkpoint-5-loss-312.086890179178-lr-0.0001-wd-1e-5-ai-1.pth'
checkpoint_path_cmr='/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/3D_all_labels_lstm_768_ld/ResNet503D_checkpoint-43-loss-87.54055601550687-lr-0.0001-wd-1e-5-ai-1.pth' # lstm 768 latent dim
ECG_TRAIN_DIR='/projects/prjs1252/CL_data/CL_3D/CL_ECG_leads_train_3d.pt'
ECG_VAL_DIR='/projects/prjs1252/CL_data/CL_3D/CL_ECG_leads_val_3d.pt'

# CHECKPOINT CMR (WITH TIME) + ALL LABELS MLP
# checkpoint_path_cmr='/home/abujalancegome/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/3D_all_labels_MLP/ResNet503D_MLP_checkpoint-25-loss-82.92884949714907-lr-0.0001-wd-1e-5-ai-1.pth'

CMR_TRAIN_DIR='/projects/prjs1252/CL_data/CL_3D/CL_cmr_tensor_train_3d.pt'
CMR_VAL_DIR='/projects/prjs1252/CL_data/CL_3D/CL_cmr_tensor_val_3d.pt'


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

            python train_cl.py --model_name ResNet50-3D \
                --ecg_model mae_vit_mediumDeep_patchX \
                --batch_size 64 \
                --cmr_train_path ${CMR_TRAIN_DIR} \
                --ecg_train_path ${ECG_TRAIN_DIR} \
                --cmr_val_path ${CMR_VAL_DIR} \
                --ecg_val_path ${ECG_VAL_DIR} \
                --checkpoint_path_cmr ${checkpoint_path_cmr} \
                --checkpoint_path_ecg ${checkpoint_path_ecg} \
                --num_outputs 76 \
                --output_dir CL_results/medium3D_all_labels_LSTM_freeze\
                --epochs 200 \
                --latent_dim 768 \
                --projection_dim 258 \
                --accum_iter ${ai} \
                --lr ${lr} \
                --weight_decay ${wd}
        done
    done
done