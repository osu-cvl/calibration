"""
model definition and utils
"""
import torch
import torch.nn as nn
import torchvision.models as models


def resnet_for_iNat2019():
    model = models.resnet18()
    model.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(in_features=model.fc.in_features,out_features=1010,bias=True))
    return model

def load_model(path):
    model = resnet_for_iNat2019()
    #model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    return model
