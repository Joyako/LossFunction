import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class AngularSoftmaxLinear(nn.Module):
    """

    Ref:
        1. SphereFace: Deep Hypersphere Embedding for Face Recognition
        https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py

    """

    def __init__(self, in_features, out_features, margin=4):
        super().__init__()
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.iter = 0
        self.cos_val = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def forward(self, input, target):
        device = input.device
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        self.weight.to(device)
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.cos_val[self.margin](cos_theta)

        theta = cos_theta.data.acos()
        k = self.margin * theta / math.pi
        k = k.floor()
        phi_theta = (-1.0) ** k * cos_m_theta - 2 * k

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1)

        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= torch.norm(input, 2, 1).view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', margin=' + str(self.margin) + ')'


# class AngularSoftmaxLoss(nn.Module):
#     def __init__(self, gamma):
#         super().__init__()
#         self.gamma = gamma
#         self.iter = 0
#         self.lamb = 1500.
#
#     def forward(self, predict, target):
#         cos_theta, phi_theta = predict
#
#         index = torch.zeros(cos_theta.size(), dtype=torch.float32)
#         index.scatter_(1, target, 1)
#         index = index.byte()
#
#         output = cos_theta
#         self.iter = self.iter + 1
#         self.lamb = max(5., 1500. / (1. + 0.1 * self.iter))
#         output[index] -= cos_theta[index] / (1 + self.lamb)
#         output[index] += phi_theta[index] / (1 + self.lamb)
#
#         logpt = F.log_softmax(output)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = logpt.data.exp()
#
#         # Focal loss
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         loss = loss.mean()
#
#         return loss

