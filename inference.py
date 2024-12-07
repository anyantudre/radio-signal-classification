import config
from dataset.dataset import SpecDataset, CLASSES
from models.classifier import SpecModel
from utils.utils import confusion_matrix, plot_confusion_matrix

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# evaluate the best weights saved by train.py on the validation set
model = SpecModel(pretrained=False)
model.load_state_dict(torch.load(config.WEIGHTS_PATH, map_location=config.DEVICE))
model.to(config.DEVICE)
model.eval()

validloader = DataLoader(SpecDataset(pd.read_csv(config.VALID_CSV)), batch_size=config.BATCH_SIZE)

y_true, y_pred, total_loss = [], [], 0.0
with torch.no_grad():
    for images, labels in validloader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        logits, loss = model(images, labels)
        total_loss += loss.item() * len(labels)
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=1).tolist())

y_true, y_pred = np.array(y_true), np.array(y_pred)
cm = confusion_matrix(y_true, y_pred, num_classes=len(CLASSES))

print(f"validation examples: {len(y_true)}")
print(f"validation loss: {total_loss / len(y_true):.4f}")
print(f"validation accuracy: {(y_true == y_pred).mean():.4f}")
for i, name in enumerate(CLASSES):
    print(f"  {name:<14} {cm[i, i]}/{cm[i].sum()} correct ({cm[i, i] / cm[i].sum():.1%})")
print("confusion matrix (rows = true, columns = predicted):")
print(cm)

os.makedirs('media', exist_ok=True)
plot_confusion_matrix(cm, title=f'{config.MODEL_NAME} on valid.csv ({len(y_true)} spectrograms)',
                      save_path='media/confusion-matrix.png')
print("saved media/confusion-matrix.png")
