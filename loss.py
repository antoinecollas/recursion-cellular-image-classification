import math
import torch
import torch.nn as nn

class ArcFaceLoss():
    def __init__(self, s=64, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.softmax = nn.Softmax()
        self.cross_entropy = nn.CrossEntropyLoss()

    def to(self, device):
        self.softmax = self.softmax.to(device)
        self.cross_entropy = self.cross_entropy.to(device)
        return self

    def __call__(self, cos_th, target):
        # cos(theta+m)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2)) #.clamp(0, 1))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m

        # if cos(theta) <= cos(theta+m) (i.e theta+m >= pi): cos(theta) - sin(pi-m)*m
        cos_th_m = torch.where(cos_th-self.th > 0, cos_th_m, cos_th - self.mm)

        one_hot = torch.zeros_like(cos_th)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        output = (one_hot * cos_th_m) + ((1.0 - one_hot) * cos_th)
        output = output * self.s
        
        output = self.softmax(output)
        output = self.cross_entropy(output, target)
        return output