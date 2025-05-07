import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Train Validation Test Split', add_help=False)

    # Main arguments
    parser.add_argument('--data_path', default='/scratch-shared/abujalancegome/CMR_data', type=str,
                        help='Path to the original data')

    parser.add_argument('--output_dir', default='/scratch-shared/abujalancegome/CMR_data', type=str,
                        help='Path to save the data splits')
    
    return parser


def main(args):
    # Load the full dataset: 10-s, 12-lead ECG, original samplig rate frequency 500 Hz
    loaded_data = torch.load(args.data_path).unsqueeze(1)

    participant_ids = pd.read_csv('labels/ids_cmr_dataset.csv', header=None)

    labels_df = pd.read_csv('labels/labels_CMR_pretrain.csv', header=None)

    print(loaded_data.shape)
    print(loaded_data.dtype)

    # Convert PyTorch tensor to NumPy for sklearn
    full_data_np = loaded_data.numpy()
    participant_ids_np = np.array(participant_ids)

    # Split into train (70%) and temp (30%) which will be further split into val and test
    train_np, temp_data, train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        full_data_np, participant_ids_np, labels_df, test_size=0.3, random_state=42)

    # Split temp into validation and test
    val_np, test_np, val_ids, test_ids, val_labels, test_labels = train_test_split(
        temp_data, temp_ids, temp_labels, test_size=0.5, random_state=42)

    # Convert back to PyTorch tensors and unsqueeze to get the correct tensor shapes
    train_data = torch.tensor(train_np)
    print("train data size:", train_data.shape)
    val_data = torch.tensor(val_np)
    print("val data size:", val_data.shape)
    test_data = torch.tensor(test_np)
    print("test data size:", test_data.shape)

    torch.save(train_data, os.path.join(args.output_dir, 'cmr_tensors_ts_ALL_train.pt'))
    torch.save(val_data, os.path.join(args.output_dir, 'cmr_tensors_ts_ALL_val.pt'))
    torch.save(test_data, os.path.join(args.output_dir, 'cmr_tensors_ts_ALL_test.pt'))

    pd.DataFrame(train_ids).to_csv(os.path.join('labels', 'ids_cmr_pretrain_train_ts_ALL.csv'), header=False, index=False)
    pd.DataFrame(val_ids).to_csv(os.path.join('labels', 'ids_cmr_pretrain_val_ts_ALL.csv'), header=False, index=False)
    pd.DataFrame(test_ids).to_csv(os.path.join('labels', 'ids_cmr_pretrain_test_ts_ALL.csv'), header=False, index=False)

    pd.DataFrame(train_labels).to_csv(os.path.join('labels', 'labels_CMR_pretr_trn_ts_ALL.csv'), header=False, index=False)
    pd.DataFrame(val_labels).to_csv(os.path.join('labels', 'labels_CMR_pretr_val_ts_ALL.csv'), header=False, index=False)
    pd.DataFrame(test_labels).to_csv(os.path.join('labels', 'labels_CMR_pretr_tst_ts_ALL.csv'), header=False, index=False)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    main(args)