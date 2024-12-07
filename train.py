import config
import trainer_api
from dataset.dataset import SpecDataset
from models.classifier import SpecModel
from utils.utils import get_train_tranform

import random

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader 


random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

# load datasets and dataset loaders
df_train = pd.read_csv(config.TRAIN_CSV)
df_valid = pd.read_csv(config.VALID_CSV)


trainloader = DataLoader(
    SpecDataset(df_train, augs=get_train_tranform()), 
    batch_size=config.BATCH_SIZE, 
    shuffle=True
)

validloader = DataLoader(
    SpecDataset(df_valid), 
    batch_size=config.BATCH_SIZE
)

# load model
model = SpecModel().to(config.DEVICE)

# training
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
history = trainer_api.fit(model, trainloader, validloader, optimizer)

best = min(history, key=lambda h: h['valid_loss'])
print(f"best epoch {best['epoch']}: valid loss {best['valid_loss']:.4f}, valid acc {best['valid_acc']:.4f}")
