import torch
import timm
from torch import nn
import torch.nn.functional as F
import numpy as np
# import pandas as pd


# from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import RandomOverSampler
# import torchvision

# from torch.utils.data import Dataset, DataLoader


# from torch.optim import Adam
# from tqdm.notebook import tqdm
# from timeit import default_timer as timer
# from pathlib import Path
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # DEVICE = 'cpu'
# print(DEVICE)

class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model_name = "mobilenetv2_050"
        model_name = "mobilenetv3_small_050"
        # self.model = timm.create_model(model_name = "resnet50", pretrained = True)
        self.model = timm.create_model(model_name = model_name, pretrained = True)


        # Register a forward hook for the layer
        self.intermediate_features = None  # Variable to store intermediate features

        def hook(module, input, output):
            self.intermediate_features = output

        # Register the hook to the layer
        self.hook_handle = self.model.blocks[5][0].conv.register_forward_hook(hook)
        # self.hook_handle = self.model.conv_head.register_forward_hook(hook)


        num_in_features = self.model.get_classifier().in_features

        new_classifier_head =  nn.Sequential(
                nn.BatchNorm1d(num_in_features),
                nn.Linear(in_features=num_in_features, out_features=512, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.4),
                nn.Linear(in_features=512, out_features= num_classes, bias=False),
                #nn.Softmax(dim=-1) //Not needed here since softmax is done inside CrossEntropyLoss(), returning pure logits
            )
    
        self.model.classifier = new_classifier_head

        # if model_name == "mobilenetv2_050":
        #     self.model.classifier = new_classifier_head
        # elif model_name == "resnet50":
        #     self.model.fc = new_classifier_head

    def forward(self,X):
        # if self.model.training:
        #     return self.model(X)
        # else:
        #     return F.softmax(self.model(X))
        return self.model(X)
    
    def cleanup_hooks(self):
        # Remove the hook after using it to avoid memory leaks
        self.hook_handle.remove()