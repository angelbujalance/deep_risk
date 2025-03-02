import torch

full_data = torch.load("full_data.pt")
print(full_data.shape)

mean, std = full_data.mean(), full_data.std()
full_data = (full_data - mean) / std

train_data = full_data[:2000,:,:]
train_data = train_data.unsqueeze(2)
print(train_data.shape)

val_data = full_data[2000:3000,:,:]
val_data = val_data.unsqueeze(2)
print(val_data.shape)

torch.save(train_data, "ECG_leads_full_pretraining_train.pt")
torch.save(val_data, "ECG_leads_full_pretraining_val.pt")