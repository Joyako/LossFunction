import torch.nn as nn
import torch
import math
from scipy.special import binom


class LargeMarginSoftmaxLinear(nn.Module):
    """

    Ref:
        1. Large-Margin Softmax Loss for Convolutional Neural Networks
        (https://arxiv.org/pdf/1612.02295.pdf)
        2. https://github.com/amirhfarzaneh/lsoftmax-pytorch/blob/master/lsoftmax.py

    For Li, the only difference between the original softmax
    loss and the L-Softmax loss lies in fyi. Thus we only need
    to compute fyi in forward and backward propagation while fj, j != yi
    is the same as the original softmax loss.

    That is to say, optimize target is fyi finally.
    fyi = (-1)**k * ||Wyi|| * ||xi|| * cos(m * theta) - 2 * k * ||Wyi|| * ||xi||
    """

    def __init__(self, margin, in_features, num_classes, beta=100, phase='train'):
        super().__init__()
        self.margin = margin
        self.W = nn.Parameter(torch.FloatTensor(in_features, num_classes))
        self.divisor = math.pi / self.margin
        self.eps = 1e-10
        self.phase = phase
        self.beta = beta
        self.scale = 0.99
        # Ref: Equation (7)
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2)))
        self.cos_powers = torch.Tensor(range(margin, -1, -2))
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers)))
        self.signs = torch.ones(margin // 2 + 1)
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta, device):
        self.C_m_2n.to(device)
        self.cos_powers.to(device)
        self.sin2_powers.to(device)
        self.signs.to(device)

        sin2_theta = 1. - cos_theta ** 2
        # cos(theta) ** (m - 2n)
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)
        # (sin(theta) ** 2) ** n
        sin2_terms = (sin2_theta.unsqueeze(1) ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = self.signs.unsqueeze(0) * self.C_m_2n.unsqueeze(0) * cos_terms * sin2_terms

        return cos_m_theta.sum(1)

    def calculate_k(self, cos_theta):
        cos = torch.clamp(cos_theta, -1 + self.eps, 1 - self.eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W.data.t())

    def forward(self, predict, target):
        device = predict.device

        if self.phase == 'train':
            assert target is not None
            self.W.to(device)
            beta = max(self.beta, 0)

            # Step1: calculate cos(theta)
            # vector w * vector x
            logit = predict.mm(self.W)
            idx = range(logit.size(0))
            logit_target = logit[idx, target]

            # cos(theta) = w * x / ||w||*||x||
            w_norm = self.W[:, target].norm(p=2, dim=0)
            x_norm = predict.norm(p=2, dim=1)
            cos_theta = logit_target / (w_norm * x_norm + self.eps)

            # Step2: calculate cos(m * theta)
            cos_m_theta = self.calculate_cos_m_theta(cos_theta, device)

            # Step3: calculate k and k is an integer that belongs to [0, m − 1]
            k = self.calculate_k(cos_theta)

            # Step4: update fyi
            fyi = w_norm * x_norm * (((-1) ** k * cos_m_theta) - 2 * k)
            fyi = (fyi + beta * logit[idx, target]) / (1. + beta)
            logit[idx, target] = fyi

            self.beta *= self.scale

            return logit

        else:
            return predict.mm(self.W.to(device))


