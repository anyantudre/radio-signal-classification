import config

import timm
from torch import nn


class SpecModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SpecModel, self).__init__()
        
        self.net = timm.create_model(config.MODEL_NAME, num_classes = 4, pretrained = pretrained, in_chans=1)
        
    def forward(self, images, labels = None):
        logits = self.net(images)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()
            return logits, loss(logits, labels)
        
        return logits
