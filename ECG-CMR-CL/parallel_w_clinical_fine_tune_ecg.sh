#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ECGFTcli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --array=0-5%6
#SBATCH --output=output/fine_tune_ecg_clinical/medium2D/%A_%a.out

# Activate your environment
source activate cmr_pretrain

# OUTPUT DIR TO SAVE MODEL
OUT_DIR='/projects/prjs1252/CL_data/checkpoints/MAE_ECG_finetuned/CL'

checkpoint="$HOME/deep_risk/mae/MAE_pretrain/checkpoint-141-loss-0.1758.pth" # medium
# CL checkpoint
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium2/ECGEncoder_checkpoint-15-loss-3.334488876071977.pth" # CL medium
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D/ECGEncoder_checkpoint-9-loss-3.60972042289781.pth" # CL 3D data
# checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_all_labels_ld_258_lstm_768/ECGEncoder_checkpoint-6-loss-3.9472279578079412.pth"
# checkpoint="$HOME/deep_risk/mae/MAE_pretrain/checkpoint-141-loss-0.1758.pth" # medium
# checkpoint='/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_all_labels_ld_128_lstm_768/ECGEncoder_checkpoint-6-loss-4.009204374419318.pth' # CL LSTM
# checkpoint='/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium2/ECGEncoder_checkpoint-15-loss-3.334488876071977.pth' # CL 2D
# checkpoint="$HOME/deep_risk/mae/MAE_pretrain/checkpoint-141-loss-0.1758.pth" # medium
# checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_all_labels/ECGEncoder_checkpoint-12-loss-3.4776823461791615.pth" # CL 3D data ALL LABELS
# checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_all_labels_MLP/ECGEncoder_checkpoint-10-loss-3.608468297087116.pth" # CL 3D data MLP
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_all_labels_LSTM_freeze/ECGEncoder_checkpoint-6-loss-3.960752699110243.pth"

# MODEL WITH BEST PERFORMANCE
checkpoint='/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium2/ECGEncoder_checkpoint-15-loss-3.334488876071977.pth' # CL 2D

endpoint_codes=("coronary_artery_disease" "atrial_fibrilation" "sudden_cardiac_death" "heart_failure" "myocardial_infarction" "cardiomyopathy")
splits=("train" "val" "test")

# Select the current endpoint code based on SLURM array task ID
current_endpoint="${endpoint_codes[$SLURM_ARRAY_TASK_ID]}"

BASE_DIR="/projects/prjs1252/CL_data/ECG_data"

# Dynamically generate file paths
for split in "${splits[@]}"; do
    # Generate file paths
    LEADS_PTH="${BASE_DIR}/ECG_balanced_${current_endpoint}_${split}_leads.pt"
    LABELS_PTH="${BASE_DIR}/ECG_balanced_${current_endpoint}_${split}_labels.csv"
    CLINICAL_PATH="${BASE_DIR}/ECG_balanced_${current_endpoint}_${split}_clinical_input.csv"

    # Store paths in variables to pass to Python script
    if [ "${split}" == "train" ]; then
        TRAIN_LEADS="${LEADS_PTH}"
        TRAIN_LABELS="${LABELS_PTH}"
        TRAIN_CLINICAL="${CLINICAL_PATH}"
    elif [ "${split}" == "val" ]; then
        VAL_LEADS="${LEADS_PTH}"
        VAL_LABELS="${LABELS_PTH}"
        VAL_CLINICAL="${CLINICAL_PATH}"
    else  # test
        TEST_LEADS="${LEADS_PTH}"
        TEST_LABELS="${LABELS_PTH}"
        TEST_CLINICAL="${CLINICAL_PATH}"
    fi
done

# 30 nans
# Grid search params
learning_rates=(1e-2 1e-3 1e-4 1e-5)
weight_decays=(1e-2 1e-3 1e-4)
accum_iter=(1 2)

# GRID SEARCH BEST PARAMS
lr=1e-4
wd=1e-4
ai=1

echo "Processing endpoint: ${current_endpoint}"
echo "With pre-trained checkpoint ${checkpoint}"
echo ""
echo ""
echo "Running with lr=$lr, weight_decay=$wd, accum_iter=$ai"

python fine_tune_ecg.py --model_name mae_vit_mediumDeep_patchX \
    --batch_size 64 \
    --train_path ${TRAIN_LEADS} \
    --val_path ${VAL_LEADS} \
    --test_path ${TEST_LEADS} \
    --train_labels_path ${TRAIN_LABELS} \
    --val_labels_path ${VAL_LABELS} \
    --test_labels_path ${TEST_LABELS} \
    --num_outputs 1 \
    --epochs 200 \
    --accum_iter ${ai} \
    --checkpoint_path ${checkpoint} \
    --lr ${lr} \
    --weight_decay ${wd} \
    --clinical_data true \
    --train_clinical_path ${TRAIN_CLINICAL} \
    --val_clinical_path ${VAL_CLINICAL} \
    --test_clinical_path ${TEST_CLINICAL}

echo "Finished processing endpoint: ${current_endpoint}"
echo "CL 3D data ALL LABELS"