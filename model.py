# imports
import torch
from torch import nn
import torchvision

# create model function
def create_mobile_net_v3(class_names):

    # get weights
    weights_mob=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT

    # create model
    model_mob=torchvision.models.mobilenet_v3_large(weights=weights_mob)

    #transforms
    transforms_mob=weights_mob.transforms()
    
    # freeze base layers except last 3 layers
    for param in model_mob.features[:-3].parameters():
        param.requires_grad=False

    # change head
    model_mob.classifier=model_mob.classifier=nn.Sequential(
    nn.Linear(in_features=960, out_features=1280, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    )
    return model_mob, transforms_mob