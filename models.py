import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from efficientnet_pytorch import EfficientNet

class CustomNN(nn.Module):
    def __init__(self, backbone, nb_classes, loss, hidden_neurons=2048, dropout=0.4):
        super(CustomNN, self).__init__()

        if backbone == 'resnet':
            self.backbone = models.resnet50(pretrained=True)
            trained_kernel = self.backbone.conv1.weight
            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
            self.backbone.conv1 = new_conv
            num_ftrs_cnn = 3*self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
            trained_kernel = self.backbone._conv_stem.weight
            self.backbone._conv_stem.in_channels = 6
            self.backbone._conv_stem.weight = torch.nn.Parameter(torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1))
            num_ftrs_cnn = 3*self.backbone._fc.in_features
            self.backbone._bn1 = nn.Identity()
            self.backbone._fc = nn.Identity()

        self.loss = loss
        if self.loss=='softmax':
            self.mlp = nn.Sequential(
                    nn.BatchNorm1d(num_ftrs_cnn),
                    nn.Dropout(dropout),
                    nn.Linear(num_ftrs_cnn, hidden_neurons),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_neurons),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_neurons, nb_classes)
                    )
        elif self.loss=='arcface':
            self.mlp = nn.Sequential(
                    nn.BatchNorm1d(num_ftrs_cnn),
                    nn.Dropout(dropout),
                    nn.Linear(num_ftrs_cnn, hidden_neurons),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_neurons)
                    )
            self.weight = nn.Parameter(torch.FloatTensor(nb_classes, hidden_neurons))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x shape: [batch, img/negative_control/positive_control, channel, h, w]
        bs = x.shape[0]
        x = x.reshape([-1, x.shape[2], x.shape[3], x.shape[4]])
        features = self.backbone(x)
        features = features.reshape([bs, -1, features.shape[1]])
        shape = int(features.shape[1]/3)
        features_imgs = features[:, 0:shape, :].mean(1)
        features_negative_controls = features[:, shape:2*shape, :].mean(1)
        features_positive_controls = features[:, 2*shape:, :].mean(1)
        features = torch.cat([features_imgs, features_negative_controls, features_positive_controls], dim=1)

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