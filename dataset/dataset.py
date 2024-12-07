import torch
from torch.utils.data import Dataset 
import numpy as np


# index i in this list is the integer label i
CLASSES = ['Squiggle', 'Narrowbanddrd', 'Noises', 'Narrowband']


class SpecDataset(Dataset):
    def __init__(self, df, augs = None):
        self.df = df.copy()
        self.augs = augs
        
        label_map = {name: i for i, name in enumerate(CLASSES)}
        self.df['labels'] = self.df.labels.map(label_map)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_pixels = np.array(row.iloc[:8192], dtype=np.float64)
        
        image = np.resize(image_pixels, (64, 128, 1)) # (h, w, c)
        label = np.array(row.labels, dtype=np.int64)
        
        image = torch.Tensor(image).permute(2, 0, 1)# (c, h, w)
        
        if self.augs is not None:
            image = self.augs(image)
            
        return image.float(), label
