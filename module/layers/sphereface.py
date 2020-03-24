import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class AngularSoftmaxLinear(nn.Module):
    """

    Ref:
        1. SphereFace: Deep Hypersphere Embedding for Face Recognition
        https://github.com/clcarwin/sphereface_pytorch/blob/master/net_sphere.py

    """

    def __init__(self, margin, in_features, num_classes):
        super().__init__()
        self.margin = margin
        self.W = nn.Parameter(torch.FloatTensor(in_features, num_classes))
        self.eps = 1e-10
        self.cos_val = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W.data.t())

    def forward(self, input):
        device = input.device

        self.W.to(device)
        w_modulus = self.W.pow(2).sum(0).pow(0.5)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        w = self.W / (w_modulus + self.eps)

        # Step1: calculate cos(theta)
        # cos(theta) = w * x / ||x||
        logit = input.mm(w)
        cos_theta = (logit / x_modulus).clamp(-1, 1)

        # Step2: calculate cos(m * theta)
        cos_m_theta = self.cos_val[self.margin](cos_theta)
        theta = cos_theta.data.acos()
        k = self.margin * theta / math.pi
        k = k.floor()
        phi_theta = (-1) ** k * cos_m_theta - 2 * k

        # Step3:
        cos_theta = cos_theta * x_modulus.view(-1, 1)
        phi_theta = phi_theta * x_modulus.view(-1, 1)

        return cos_theta, phi_theta


class AngularSoftmaxLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.iter = 0
        self.lamb = 1500.

    def forward(self, predict, target):
        cos_theta, phi_theta = predict

        index = torch.zeros(cos_theta.size(), dtype=torch.float32)
        index.scatter_(1, target, 1)
        index = index.byte()

        output = cos_theta
        self.iter = self.iter + 1
        self.lamb = max(5., 1500. / (1. + 0.1 * self.iter))
        output[index] -= cos_theta[index] / (1 + self.lamb)
        output[index] += phi_theta[index] / (1 + self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        # Focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss

