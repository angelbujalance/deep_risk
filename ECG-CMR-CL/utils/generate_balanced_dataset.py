import argparse
import torch
import os
import pandas as pd
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('Generate the CL dataset based on the ECG and the CMR data paths',
                                     add_help=False)

    parser.add_argument('--ecg_data_dir', default='/home/abujalancegome/deep_risk/data', type=str, help='Path to the ECG data directory')
    parser.add_argument('--output_dir', default='/projects/prjs1252/CL_data/ECG_data',
                        type=str, help='The output directory to save the ECG dataset for fine-tunning')

    return parser


def main(args):

    # Get the directory of the current Python file
    initial_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory
    parent_dir = "/gpfs"

    # Change the working directory to the parent directory
    os.chdir(parent_dir)

    print(f"Initial working directory: {initial_file_dir}")
    print(f"Current working directory: {os.getcwd()}")

    # Load ECG tensor and labels
    train_labels = pd.read_csv(os.path.join(args.output_dir, "ECG_fine_tune_train_labels.csv"), header=None)
    print(len(train_labels))
    train_ECG_leads = torch.load(os.path.join(args.output_dir,"ECG_leads_train"))

    # select the indexes with at least one positive case
    positive_mask = train_labels.sum(axis=1) > 0
    negative_mask = ~positive_mask

    positive_indices = np.where(positive_mask)[0]
    negative_indices = np.where(negative_mask)[0]

    print(positive_indices[:10])
    print(len(positive_mask), len(positive_indices))

    # balance the dataset, removing extra negatives indices
    np.random.shuffle(negative_indices)
    negative_indices = negative_indices[:len(positive_indices)]

    # shuffle the positive and negative cases
    train_indices = np.append(positive_indices, negative_indices)
    np.random.shuffle(train_indices)

    # get balanced data and labels
    balanced_train_labels = train_labels.iloc[train_indices].reset_index(drop=True)
    del train_labels
    balanced_train_ECG_leads = train_ECG_leads[train_indices]
    del train_ECG_leads

    torch.save(balanced_train_ECG_leads, os.path.join(args.output_dir, "ECG_balanced_train_leads.pt"))
    balanced_train_labels.to_csv(os.path.join(args.output_dir, "ECG_balanced_train_labels.csv"), header=False, index=False)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)