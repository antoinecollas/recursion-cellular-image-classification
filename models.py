import sys

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision import models

class MultiMLP(nn.Module):
    def __init__(self, nb_mlp, size_input, size_output, hidden_neurons, dropout):
        super(MultiMLP, self).__init__()

        self.bn_0 = nn.ModuleList([nn.BatchNorm1d(size_input) for i in range(nb_mlp)])
        self.weight_fc0 = nn.Parameter(torch.FloatTensor(1, nb_mlp, hidden_neurons, size_input))
        init.kaiming_uniform_(self.weight_fc0, a=math.sqrt(5))
        self.bias_fc0 = nn.Parameter(torch.FloatTensor(1, nb_mlp, hidden_neurons, 1))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_fc0)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias_fc0, -bound, bound)

        self.bn_1 = nn.ModuleList([nn.BatchNorm1d(hidden_neurons) for i in range(nb_mlp)])
        self.weight_fc1 = nn.Parameter(torch.FloatTensor(1, nb_mlp, size_output, hidden_neurons))
        init.kaiming_uniform_(self.weight_fc1, a=math.sqrt(5))
        self.bias_fc1 = nn.Parameter(torch.FloatTensor(1, nb_mlp, size_output, 1))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.bias_fc1)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias_fc1, -bound, bound)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        features = x[:, None, :]
        output = features.repeat(1, 4, 1)

        temp = output.new_zeros(output.shape)
        for i in range(len(self.bn_0)):
            temp[:, i, :] = self.bn_0[i](output[:, i, :].squeeze(1))
        output = temp
        output = output[:, :, :, None]
        output = self.dropout(output)
        output = self.weight_fc0 @ output + self.bias_fc0
        output = F.relu(output)
        output = output.squeeze(3)

        temp = output.new_zeros(output.shape)
        for i in range(len(self.bn_1)):
            temp[:, i, :] = self.bn_1[i](output[:, i, :].squeeze(1))
        output = temp
        output = output[:, :, :, None]
        output = self.dropout(output)
        output = self.weight_fc1 @ output + self.bias_fc1
        output = output.squeeze(3)

        return output

class CustomNN(nn.Module):
    def __init__(self, pretrained, plates_groups, hidden_neurons=1024, dropout=0.3):
        super(CustomNN, self).__init__()

        nb_plates, nb_classes_per_plate = plates_groups.shape
        self.plates_groups = plates_groups.reshape(-1)
        self.permutation = np.zeros(len(self.plates_groups), dtype=np.int32)
        for i, sirna in enumerate(self.plates_groups):
            self.permutation[sirna] = i

        self.base_nn = models.resnet50(pretrained=pretrained)
        trained_kernel = self.base_nn.conv1.weight
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
        self.base_nn.conv1 = new_conv
        num_ftrs_cnn = 3*self.base_nn.fc.in_features
        self.base_nn.fc = nn.Identity()

        self.multi_mlp = MultiMLP(nb_mlp=nb_plates, size_input=num_ftrs_cnn, \
            size_output=nb_classes_per_plate, hidden_neurons=hidden_neurons, dropout=dropout)

        self.logsoftmax = nn.LogSoftmax(dim=2)

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

        output = self.multi_mlp(features)

        output = self.logsoftmax(output)
        output = output.reshape(output.shape[0], -1)

        res = output[:, self.permutation]

        return res

class DummyClassifier():
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
    
    def __call__(self, x):
        bs = x.shape[0]
        output = torch.zeros((bs, self.nb_classes)).random_(-10000, 10000)/10000
        return output