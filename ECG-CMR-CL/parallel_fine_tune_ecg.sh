#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ECGFTcli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --array=0%1
#SBATCH --output=output/fine_tune_ecg/medium3D_norm_by_patients_LVEF_no_CL/%A_%a.out

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

checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_MLP_10_lables/ECGEncoder_checkpoint-32-loss-3.410451309180554.pth"
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_MLP_10_lables_lr_best/ECGEncoder_checkpoint-13-loss-3.443430679815787.pth"
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_triple_CL_2/ECGEncoder_checkpoint-13-loss-2.729943737571622.pth"

# MODEL WITH BEST PERFORMANCE
# checkpoint='/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium2/ECGEncoder_checkpoint-15-loss-3.334488876071977.pth' # CL 2D
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_triple_CL/ECGEncoder_checkpoint-45-loss-0.38694399615956676.pth" # TRIPLE CL DIASTOLE + SYSTOLE
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_triple_CL_2/ECGEncoder_checkpoint-24-loss-0.894310935963819.pth"

checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_triple_CL_2/ECGEncoder_checkpoint-13-loss-2.729943737571622.pth"
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_triple_CL/ECGEncoder_checkpoint-10-loss-2.4915894593721553.pth"

# MODEL NORMALIZATION BY PARTICIPANT
checkpoint="/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium3D_norm_by_patients/ECGEncoder_checkpoint-28-loss-3.1566538295628113.pth"

# MODEL NORMALIZATION BY PARTICIPANT NO CL

checkpoint="/home/abujalancegome/deep_risk/mae/MAE_pretrain/medium/checkpoint-226-loss-0.0210.pth"

checkpoint='/home/abujalancegome/deep_risk/ECG-CMR-CL/CL_results/medium2/ECGEncoder_checkpoint-15-loss-3.334488876071977.pth' # CL 2D


endpoint_codes=("coronary_artery_disease" "atrial_fibrilation" "sudden_cardiac_death" "heart_failure" "myocardial_infarction" "cardiomyopathy")

endpoint_codes=("myocardial_infarction")

splits=("train" "val" "test")

# Select the current endpoint code based on SLURM array task ID
current_endpoint="${endpoint_codes[$SLURM_ARRAY_TASK_ID]}"

BASE_DIR="/projects/prjs1252/CL_data/ECG_data"

# Dynamically generate file paths
for split in "${splits[@]}"; do
    # Generate file paths
    LEADS_PTH="${BASE_DIR}/ECG_balanced_${current_endpoint}_${split}_leads.pt"
    LABELS_PTH="${BASE_DIR}/ECG_balanced_${current_endpoint}_${split}_labels.csv"
    UNBAL_LEADS_PTH="${BASE_DIR}/ECG_unbalanced_${split}_leads.pt"
    UNBAL_LABELS_PTH="${BASE_DIR}/ECG_unbalanced_${current_endpoint}_${split}_labels.csv"

    # Store paths in variables to pass to Python script
    if [ "${split}" == "train" ]; then
        TRAIN_LEADS="${LEADS_PTH}"
        TRAIN_LABELS="${LABELS_PTH}"
    elif [ "${split}" == "val" ]; then
        VAL_LEADS="${LEADS_PTH}"
        VAL_LABELS="${LABELS_PTH}"
    else  # test
        TEST_LEADS="${LEADS_PTH}"
        TEST_LABELS="${LABELS_PTH}"
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

# LV_end_systolic, LV_ejection_fraction
TRAIN_LEADS="/projects/prjs1252/CL_data/ECG_data/normalized/ECG_balanced_LV_ejection_fraction_train_leads.pt"
TRAIN_LABELS="/projects/prjs1252/CL_data/ECG_data/normalized/ECG_balanced_LV_ejection_fraction_train_labels_binary.csv"
VAL_LEADS="/projects/prjs1252/CL_data/ECG_data/normalized/ECG_balanced_LV_ejection_fraction_val_leads.pt"
VAL_LABELS="/projects/prjs1252/CL_data/ECG_data/normalized/ECG_balanced_LV_ejection_fraction_val_labels_binary.csv"
TEST_LEADS="/projects/prjs1252/CL_data/ECG_data/normalized/ECG_balanced_LV_ejection_fraction_test_leads.pt"
TEST_LABELS="/projects/prjs1252/CL_data/ECG_data/normalized/ECG_balanced_LV_ejection_fraction_test_labels_binary.csv"

python fine_tune_ecg.py --model_name mae_vit_mediumDeep_patchX \
    --batch_size 64 \
    --train_path ${TRAIN_LEADS} \
    --val_path ${VAL_LEADS} \
    --test_path ${TEST_LEADS} \
    --train_labels_path ${TRAIN_LABELS} \
    --val_labels_path ${VAL_LABELS} \
    --test_labels_path ${TEST_LABELS} \
    --num_outputs 1 \
    --output_dir output/fine_tune_ecg/medium3D_norm_by_patients_LVEF_no_CL_binary \
    --epochs 200 \
    --patience 20 \
    --accum_iter ${ai} \
    --checkpoint_path ${checkpoint} \
    --lr ${lr} \
    --weight_decay ${wd} \

echo "Finished processing endpoint: ${current_endpoint}"
echo "CL 3D data ALL LABELS"