import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class TwoSitesNN(nn.Module):
    def __init__(self, pretrained, nb_classes, size_features=1024, dropout=0.3):
        super(TwoSitesNN, self).__init__()

        self.base_nn = models.resnet50(pretrained=pretrained)
        trained_kernel = self.base_nn.conv1.weight
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
        self.base_nn.conv1 = new_conv
        num_ftrs_cnn = 3*self.base_nn.fc.in_features
        self.base_nn.fc = nn.Identity()

        self.mlp = nn.Sequential(
                nn.BatchNorm1d(num_ftrs_cnn),
                nn.Dropout(dropout),
                nn.Linear(num_ftrs_cnn, size_features),
                nn.ReLU(),
                nn.BatchNorm1d(size_features),
                nn.Dropout(dropout),
                nn.Linear(size_features, nb_classes)
                )

    def forward(self, x):
        # x shape: [batch, img/negative_control/positive_control, channel, h, w]
        bs = x.shape[0]
        x = x.reshape([-1, x.shape[2], x.shape[3], x.shape[4]])
        features = self.base_nn(x)
        features = features.reshape([bs, -1, features.shape[1]])
        shape = int(features.shape[1]/3)
        features_imgs = features[:, 0:shape, :].mean(1)
        features_negative_controls = features[:, shape:2*shape, :].mean(1)
        features_positive_controls = features[:, 2*shape:, :].mean(1)
        features = torch.cat([features_imgs, features_negative_controls, features_positive_controls], dim=1)

        output = self.mlp(features)

        return output

class DummyClassifier():
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
    
    def __call__(self, x):
        bs = x.shape[0]
        output = torch.zeros((bs, self.nb_classes)).random_(-10000, 10000)/10000
        return output