import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class TwoSitesNN(nn.Module):
    def __init__(self, pretrained, nb_classes):
        super(TwoSitesNN, self).__init__()

        self.base_nn = models.resnet18(pretrained=pretrained)
        trained_kernel = self.base_nn.conv1.weight
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
        self.base_nn.conv1 = new_conv
        num_ftrs = self.base_nn.fc.in_features
        self.base_nn.fc = nn.Identity()

        self.classifier = torch.nn.Linear(num_ftrs, nb_classes)

    def forward(self, x):
        # x shape: [batch, site, channel, h, w]

        site_1 = x[:, 0, :, :, :].squeeze(1)
        size_site_1 = site_1.shape[0]
        site_2 = x[:, 1, :, :, :].squeeze(1)
        size_site_2 = site_2.shape[0]
        temp = torch.cat([site_1, site_2])
        features = self.base_nn(temp)
        features_site_1 = features[:size_site_1]
        features_site_2 = features[size_site_1:]
        features = (features_site_1+features_site_2)/2
        output = self.classifier(features)

        return output

class DummyClassifier():
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
    
    def __call__(self, x):
        bs = x.shape[0]
        output = torch.zeros((bs, self.nb_classes)).random_(-10000, 10000)/1000
        return output