#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=CMR_pret
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=output/cmr_pretrain_FULL_DATA_3D.out

# Activate your environment
source activate cmr_pretrain

# DATA DIRECTORIES
DATA_DIR='/projects/prjs1252/CL_data/cmr_tensors_ts_all_train.pt'
VAL_DIR='/projects/prjs1252/CL_data/cmr_tensors_ts_all_val.pt'
TEST_DIR='/projects/prjs1252/CL_data/cmr_tensors_ts_all_test.pt'

# LABELS
TRAIN_LABELS_PTH="$HOME/deep_risk/ECG-CMR-CL/cmr_pretrain/labels/labels_CMR_pretr_trn.csv"
VAL_LABELS_PTH="$HOME/deep_risk/ECG-CMR-CL/cmr_pretrain/labels/labels_CMR_pretr_val.csv"
TEST_LABELS_PTH="$HOME/deep_risk/ECG-CMR-CL/cmr_pretrain/labels/labels_CMR_pretr_tst.csv"


# LOAD CHECKPOINTS
# checkpoint="$HOME/deep_risk/ECG-CMR-CL/cmr_pretrain/CMR_pretrain/ResNet50_checkpoint-196-loss-415.7585052490234-lr-1e-06-wd-1e-5-ai-1.pth"

export CUDA_LAUNCH_BLOCKING=1

# 30 nans
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
# test_dir2/lr=${lr}_wd=${wd}_acit=${ai}
accum_iter=(1)

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

            python pretrain_cmr.py --model_name ResNet50-3D \
                --batch_size 96 \
                --train_path ${DATA_DIR} \
                --val_path ${VAL_DIR} \
                --test_path ${TEST_DIR} \
                --train_labels_path ${TRAIN_LABELS_PTH} \
                --val_labels_path ${VAL_LABELS_PTH} \
                --test_labels_path ${TEST_LABELS_PTH} \
                --num_outputs 3 \
                --epochs 200 \
                --output_dir CMR_pretrain/3D \
                --accum_iter ${ai} \
                --lr ${lr} \
                --weight_decay ${wd}
        done
    done
done
