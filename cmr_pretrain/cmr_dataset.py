import torch
# import pandas as pd

class CMRDataset(Dataset):
    def __init__(self, data_path, labels_path=None,
                 train=True, transform=None):
        self.data = torch.load(data_path, map_location=torch.device('cpu'))
        self.transform = transform

        if labels_path: # decide if labels are a torch file or a cdv
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'))
            # self.labels = pd.read_csv(labels_path) 
        else:
            raise Exception("Labels not provided. Not implemented self-supervised training.")
            

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        cmr_data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            cmr_data = self.transform(cmr_data)

        return cmr_data, label 

