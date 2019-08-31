import math
import torch
import torch.nn as nn

class ArcFaceLoss():
    def __init__(self, s, m):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.cross_entropy = nn.CrossEntropyLoss()

    def to(self, device):
        self.cross_entropy = self.cross_entropy.to(device)
        return self

    def __call__(self, cos_th, target):
        th = torch.acos(cos_th).clamp(0, math.pi)
        cos_th_m = torch.cos(th+self.m).clamp(-1, 1)
        cos_th_m = torch.where(cos_th-self.th > 0, cos_th_m, cos_th - self.mm)
        one_hot = torch.zeros_like(cos_th)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        output = (one_hot * cos_th_m) + ((1.0 - one_hot) * cos_th)
        output = output * self.s
        
        output = self.cross_entropy(output, target)
        return output