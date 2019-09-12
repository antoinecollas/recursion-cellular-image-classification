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
        num_ftrs_cnn = 3*self.base_nn.fc.in_features
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

    def forward(self, x, test_mode=False):
        # x shape: [batch, img/negative_control/positive_control, channel, h, w]
        if test_mode:
            bs = x.shape[0]
            assert bs == 1
            x = x.squeeze(0)
            features_img = self.base_nn(x[:2])
            features_negative_controls = self.base_nn(x[2:4])
            features_positive_controls = self.base_nn(x[4:])
            feature_img = features_img.mean(0, keepdim=True)
            feature_negative_control = features_negative_controls.mean(0, keepdim=True)
            feature_positive_control = features_positive_controls.mean(0, keepdim=True)
            features = torch.cat([feature_img, feature_negative_control, feature_positive_control], dim=1)
        else:
            bs = x.shape[0]
            img = x[:, 0, :, :, :].squeeze(1)
            img_negative_control = x[:, 1, :, :, :].squeeze(1)
            img_positive_control = x[:, 2, :, :, :].squeeze(1)
            temp = torch.cat([img, img_negative_control, img_positive_control])
            features = self.base_nn(temp)
            features_img = features[:bs]
            features_negative_control = features[bs:2*bs]
            features_positive_control = features[2*bs:3*bs]
            features = torch.cat([features_img, features_negative_control, features_positive_control], dim=1)

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