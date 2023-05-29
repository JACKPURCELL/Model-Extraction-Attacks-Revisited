import torch
from torch.utils.data import Dataset

class advdataset(Dataset):
    def __init__(self, x, adv_x, adv_soft_label,adv_hapi_label):
        self.x = x
        self.adv_x = adv_x
        self.adv_soft_label = adv_soft_label
        self.adv_hapi_label = adv_hapi_label
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.adv_x[index]
        z = self.adv_soft_label[index]
        c = self.adv_hapi_label[index]
        return x, y, z, c 
    
    def __len__(self):
        return len(self.x)
