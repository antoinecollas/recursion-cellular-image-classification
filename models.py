import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class TwoSitesNN(nn.Module):
    def __init__(self, pretrained, nb_classes, loss, size_features=1024, dropout=0.3):
        super(TwoSitesNN, self).__init__()

        self.base_nn = models.resnet50(pretrained=pretrained)
        trained_kernel = self.base_nn.conv1.weight
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
        self.base_nn.conv1 = new_conv
        num_ftrs_cnn = 2*self.base_nn.fc.in_features
        self.base_nn.fc = nn.Identity()

        self.loss = loss
        if self.loss=='softmax':
            self.mlp = nn.Sequential(
                    nn.BatchNorm1d(num_ftrs_cnn),
                    nn.Dropout(dropout),
                    nn.Linear(num_ftrs_cnn, size_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(size_features),
                    nn.Dropout(dropout),
                    nn.Linear(size_features, nb_classes)
                    )
        elif self.loss=='arcface':
            self.mlp = nn.Sequential(
                    nn.BatchNorm1d(num_ftrs_cnn),
                    nn.Dropout(dropout),
                    nn.Linear(num_ftrs_cnn, size_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(size_features)
                    )
            self.weight = nn.Parameter(torch.FloatTensor(nb_classes, size_features))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x shape: [batch, site/control, channel, h, w]

        site_1 = x[:, 0, :, :, :].squeeze(1)
        size_site_1 = site_1.shape[0]
        site_2 = x[:, 1, :, :, :].squeeze(1)
        size_site_2 = site_2.shape[0]
        control_site_1 = x[:, 2, :, :, :].squeeze(1)
        size_control_site_1 = control_site_1.shape[0]
        control_site_2 = x[:, 3, :, :, :].squeeze(1)
        size_control_site_2 = control_site_2.shape[0]
        temp = torch.cat([site_1, site_2, control_site_1, control_site_2])
        features = self.base_nn(temp)
        features_site_1 = features[:size_site_1]
        features_site_2 = features[size_site_1:size_site_1+size_site_2]
        features_control_site_1 = features[size_site_1+size_site_2:size_site_1+size_site_2+size_control_site_1]
        features_control_site_2 = features[size_site_1+size_site_2+size_control_site_1:]
        features = (features_site_1+features_site_2)/2
        features_control = (features_control_site_1+features_control_site_2)/2
        features = torch.cat([features, features_control], dim=1)

        if self.loss=='softmax':
            output = self.mlp(features)
        elif self.loss=='arcface':
            output = self.mlp(features)
            output = F.linear(F.normalize(output), F.normalize(self.weight)).clamp(-1, 1)

        return output

class DummyClassifier():
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
    
    def __call__(self, x):
        bs = x.shape[0]
        output = torch.zeros((bs, self.nb_classes)).random_(-10000, 10000)/10000
        return output