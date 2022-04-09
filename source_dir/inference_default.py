import io
import json
import logging
import os
import numpy as np
from PIL import Image
import requests
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def net():
    '''
    Instantiates pre-trained ResNet18 model and adds classification layer to end of model.
    '''
    model = models.resnet18(pretrained=False)
    
    num_features = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_features, 1))

    return model


def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model