import argparse
import torch
import os
import pandas as pd
import numpy as np
import re

def get_args_parser():
    parser = argparse.ArgumentParser('Generate the CL dataset based on the ECG and the CMR data paths',
                                     add_help=False)

    parser.add_argument('--ecg_data_dir', default='/home/abujalancegome/deep_risk/data', type=str, help='Path to the ECG data directory')
    parser.add_argument('--output_dir', default='/home/abujalancegome/deep_risk/data',
                        type=str, help='The output directory to save the ECG dataset for fine-tunning')

    return parser


def align_ECG_data_and_pheno_data(pheno_data: pd.DataFrame, split: str, args) -> np.array:

    os.chdir(args.ecg_data_dir)

    # filter the IDs if not in ECG path
    ECG_ids = torch.load(f"ECG_ids_{split}.pt").numpy().astype(int)

    # load the ECG tensors
    ECG_leads = torch.load(f"ECG_leads_{split}.pt").numpy()

    # Find which IDs from ECG_ids_train are missing in pheno_data
    missing_ids = [id_ for id_ in ECG_ids if id_ not in pheno_data["f.eid"].values]
    print(f"Missing IDs in {split} set: {missing_ids}")

    # Get the positions of missing IDs in ECG_ids_train
    missing_positions = [np.where(ECG_ids == missing_id)[0][0] for missing_id in missing_ids]
    print(f"Positions of missing IDs in ECG_ids_{split}: {missing_positions}")

    # Extract the missing positions in the ECG data and the IDs tensors
    # Remove the elements at the missing positions
    mask = np.ones(len(ECG_ids), dtype=bool)
    mask[missing_positions] = False
    ECG_leads = ECG_leads[mask]
    ECG_leads = torch.from_numpy(ECG_leads)
    ECG_ids = ECG_ids[mask]

    os.chdir("/gpfs")
    torch.save(ECG_leads, os.path.join(args.output_dir, f"ECG_leads_{split}"))
    torch.save(ECG_ids, os.path.join(args.output_dir, f"ECG_ids_{split}"))
    del ECG_leads, missing_positions, missing_ids

    return ECG_ids

def main(args):

    # Get the directory of the current Python file
    initial_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory
    parent_dir = "/gpfs"

    # Change the working directory to the parent directory
    os.chdir(parent_dir)

    print(f"Initial working directory: {initial_file_dir}")
    print(f"Current working directory: {os.getcwd()}")

    # Get all the diagnosis columns
    pheno_data = pd.read_csv('work2/0/aus20644/data/ukbiobank/phenotypes/ukb678882.tab.gz',
                        sep='\t', compression='gzip',
                        nrows=0
                        ) 

    # diagnostic columns corresponding to clinical aspects such as sex or smoking
    diagn_cols = ['f.31.0.0', 'f.21003.2.0', 'f.20116.2.0', 'f.4080.2.0',
                  'f.23400.0.0', 'f.23406.0.0', 'f.23405.0.0']
    diagn_cols.append('f.eid')

    print(diagn_cols)

    # Load the TSV file (gzip compressed) with the diagnosis columnns
    pheno_data = pd.read_csv('work2/0/aus20644/data/ukbiobank/phenotypes/ukb678882.tab.gz',
                        sep='\t', compression='gzip',
                        usecols=diagn_cols,
                        )
    
    print("pheno_data", pheno_data)
    

    ECG_ids_train = align_ECG_data_and_pheno_data(pheno_data, split="train", args=args)
    ECG_ids_val = align_ECG_data_and_pheno_data(pheno_data, split="val", args=args)
    ECG_ids_test = align_ECG_data_and_pheno_data(pheno_data, split="test", args=args)

    # # relevant clinical codes 
    # endpoint_codes = {
    #      "clinical_codes": diagn_cols,
    # }

    # for endpoint, codes in endpoint_codes.items():
    #     pheno_data[endpoint] = pheno_data.apply(
    #         lambda row: any(code in row.values for code in codes), axis=1
    #     )

    endpoint_labels = pheno_data
    # endpoint_labels = endpoint_labels.replace({False: 0, True: 1})

    endpoint_labels["f.eid"] = pheno_data['f.eid'].astype(int)

    train_endpoint_labels = endpoint_labels[endpoint_labels.loc[:, "f.eid"].isin(ECG_ids_train)]
    print(train_endpoint_labels.shape)

    print("train_endpoint_labels.head()")
    print(train_endpoint_labels.head())

    # Count rows with at least one NaN
    nan_rows_count = train_endpoint_labels.isna().any(axis=1).sum()

    # Total number of rows
    total_rows = len(train_endpoint_labels)

    # Percentage of rows with NaNs
    nan_percentage = (nan_rows_count / total_rows) * 100

    print(f"Rows with NaNs: {nan_rows_count}")
    print(f"Percentage of rows with NaNs: {nan_percentage:.2f}%")
    exit()

    val_endpoint_labels = endpoint_labels[endpoint_labels.loc[:, "f.eid"].isin(ECG_ids_val)]
    print(val_endpoint_labels.shape)

    test_endpoint_labels = endpoint_labels[endpoint_labels.loc[:, "f.eid"].isin(ECG_ids_test)]
    print(test_endpoint_labels.shape)

    print("endpoint_labels.shape[0]", endpoint_labels.shape[0])

    print("ECG_ids_test", ECG_ids_test[:10])
    train_endpoint_labels = train_endpoint_labels.set_index("f.eid").loc[ECG_ids_train]
    print(train_endpoint_labels.head())
    val_endpoint_labels = val_endpoint_labels.set_index("f.eid").loc[ECG_ids_val]
    test_endpoint_labels = test_endpoint_labels.set_index("f.eid").loc[ECG_ids_test]

    train_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_train_clinical_input.csv"), index=False, header=False)
    val_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_val_clinical_input.csv"), index=False, header=False)
    test_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_test_clinical_input.csv"), index=False, header=False)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)