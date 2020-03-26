import torch.nn as nn
import torch
import torch.nn.functional as F


class CosFaceLinear(nn.Module):
    """Large margin cosine distance
    Ref.
        https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py
    """
    def __init__(self, inplanes, planes, s=30.0, m=0.40):
        super(CosFaceLinear, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.s = s
        self.m = m
        self.eps = 1e-8
        self.weight = nn.Parameter(torch.FloatTensor(inplanes, planes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        device = input.device
        self.weight.to(device)
        # cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        logit = input.mm(self.weight)
        x = torch.norm(input, 2, 1)
        w = torch.norm(self.weight, 2, 1)
        cos_theta = logit / torch.ger(x, w).clamp(self.eps)

        one_hot = torch.zeros(cos_theta.size()).to(device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = self.s * (cos_theta - one_hot * self.m)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'inplanes=' + str(self.inplanes) \
               + ', planes=' + str(self.planes) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
