import torch
from torch.utils.data import Dataset


class CLDataset(Dataset):
    def __init__(self, cmr_data_path, ecg_data_path,
                 train=True, cmr_transform=None, ecg_transform=None):
        self.cmr_data = torch.load(cmr_data_path, map_location=torch.device('cpu'))
        print("cmr_data_shape:", self.cmr_data.shape)
        self.cmr_transform = cmr_transform
        self.cmr_data_len = self.cmr_data.shape[0]

        self.ecg_data = torch.load(ecg_data_path, map_location=torch.device('cpu'))
        self.ecg_transform = ecg_transform
        print("ecg_data_shape:", self.ecg_data.shape)
        self.ecg_data_len = self.ecg_data.shape[0]

        assert self.cmr_data_len == self.ecg_data_len, "The length of the CMR dataset"
        " and the ECG dataset does not match."

    def __len__(self) -> int:
        return self.cmr_data_len

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        cmr_data = self.cmr_data[idx]
        ecg_data = self.ecg_data[idx]

        if self.cmr_transform:
            cmr_data = self.cmr_transform(cmr_data)

        if self.ecg_transform:
            ecg_data = self.ecg_transform(ecg_data)

        return cmr_data, ecg_data 

