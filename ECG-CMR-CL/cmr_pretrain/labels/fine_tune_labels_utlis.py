import os
import pandas as pd
import re


# Get the directory of the current Python file
initial_file_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
root_dir = '/gpfs'

# Change the working directory to the parent directory
os.chdir(root_dir)

print(f"Initial working directory: {initial_file_dir}")
print(f"Current working directory: {os.getcwd()}\n")

# Load the TSV file (gzip compressed)
pheno_data = pd.read_csv('work2/0/aus20644/data/ukbiobank/phenotypes/ukb678882.tab.gz',
                       sep='\t', compression='gzip', nrows=1,) # usecols=['f.eid', '41270']

# retrieve phenotypes of cardiac and aortic structure and function 
pheno_fields = [f for f in pheno_data.columns if re.search('f.241\d\d.2', f)]


print(pheno_fields)
print("The number of CMR phenotypes is:", len(pheno_fields))
assert len(pheno_fields) == 82

# we select the first 1000 rows as we can just test the code with an small dataset
# ids_df = pd.read_csv('/gpfs/home3/abujalancegome/patients_w_ecg_cmr.txt', names=['f.eid'])

# print("ids_df.shape", ids_df.shape)

# append the id field to match with the patients data
pheno_fields.append('f.eid')

pheno_data = pd.read_csv('work2/0/aus20644/data/ukbiobank/phenotypes/ukb678882.tab.gz',
                       sep='\t', compression='gzip', usecols=pheno_fields) # usecols=['f.eid', '41270']

# print(pheno_data.head())

# Display the result
print("pheno_data.head()")
print(pheno_data.head())

pheno_data.to_csv('/home/abujalancegome/deep_risk/cmr_pretrain/labels/all_labels_CMR_pretrain.csv')
