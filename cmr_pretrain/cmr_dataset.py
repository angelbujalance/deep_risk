import torch
from torch.utils.data import Dataset
import pandas as pd


class CMRDataset(Dataset):
    def __init__(self, data_path, labels_path=None,
                 train=True, transform=None):
        self.data = torch.load(data_path, map_location=torch.device('cpu'))
        self.transform = transform

        if labels_path: # decide if labels are a torch file or a cdv
            if labels_path.endswith(".csv"):
                labels = pd.read_csv(labels_path, header=None)
                self.labels = torch.tensor(labels.values, dtype=torch.half)
            else:
                self.labels = torch.load(labels_path, map_location=torch.device('cpu'), weights_only=False, dtype=torch.half)
            # self.labels = pd.read_csv(labels_path) 
        else:
            raise Exception("Labels not provided. Not implemented self-supervised training.")
            

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        cmr_data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            cmr_data = self.transform(cmr_data)

        return cmr_data, label 

