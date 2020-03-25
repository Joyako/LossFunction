import torch.nn as nn
import math
import torch

cfg = {
    # width, depth, resolution, dropout
    "B0": [1.0, 1.0, 224, 0.2],
    "B1": [1.0, 1.1, 240, 0.2],
    "B2": [1.1, 1.2, 260, 0.3],
    "B3": [1.2, 1.4, 300, 0.3],
    "B4": [1.4, 1.8, 380, 0.4],
    "B5": [1.6, 2.2, 456, 0.4],
    "B6": [1.8, 2.6, 528, 0.5],
    "B7": [2.0, 3.1, 600, 0.5],
    "L2": [4.3, 5.3, 800, 0.5]
}

# Default parameters.
Stage = {
    # kernel_size, inplanes, outplanes, num_repeat_per_layer,expand_ratio
    "2": {"kernel_size": 3, "inplanes": 32, "outplanes": 16, "stride": 1, "layers": 1, "expand_ratio": 1, },
    "3": {"kernel_size": 3, "inplanes": 16, "outplanes": 24, "stride": 2, "layers": 2, "expand_ratio": 6, },
    "4": {"kernel_size": 5, "inplanes": 24, "outplanes": 40, "stride": 2, "layers": 2, "expand_ratio": 6, },
    "5": {"kernel_size": 3, "inplanes": 40, "outplanes": 80, "stride": 2, "layers": 3, "expand_ratio": 6, },
    "6": {"kernel_size": 5, "inplanes": 80, "outplanes": 112, "stride": 1, "layers": 3, "expand_ratio": 6, },
    "7": {"kernel_size": 5, "inplanes": 112, "outplanes": 192, "stride": 2, "layers": 4, "expand_ratio": 6, },
    "8": {"kernel_size": 3, "inplanes": 192, "outplanes": 320, "stride": 1, "layers": 1, "expand_ratio": 6, },
}


class _Swish(torch.autograd.Function):
    """
    f(x) = x / (1 + exp(-x))
    z(x) = 1 / (1 + exp(-x))
    df(x) / dx = z(x) + x * exp(x) / (1 + z(x)) ** 2
               = z(x) * (1 + x * (1 - z(x)))
    """

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):

    def forward(self, x):
        return _Swish.apply(x)


def _round_to_multiple_of(val, width, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    val = val * width
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


class SeModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x).squeeze(3).squeeze(2)
        y = self.fc(y).unsqueeze(2).unsqueeze(3)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):

    def __init__(self, inp, oup, kernel_size, stride, expand_ratio,
                 bn_momentum=0.1, activate="swish"):
        super(MBConvBlock, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        padding = (kernel_size - 1) // 2
        hidden_dim = inp * expand_ratio
        self.apply_residual = (inp == oup and stride == 1)
        if activate == "swish":
            self.activate = Swish()
        elif activate == "relu":
            self.activate = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum=bn_momentum),
            self.activate,

            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding,
                      stride=stride, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum=bn_momentum),
            self.activate,

            # Squeeze-and-excitation
            SeModule(hidden_dim),

            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup, momentum=bn_momentum)
        )

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


class EfficientNet(nn.Module):
    def __init__(self, name, num_classes):
        super(EfficientNet, self).__init__()
        self.width, self.depth, self.resolution, self.dropout = cfg[name]
        self.divisor = 8
        self.stage = Stage
        planes = _round_to_multiple_of(32, self.width, self.divisor)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        block = MBConvBlock
        self.stage2 = self._make_layer(block, self.stage["2"])
        self.stage3 = self._make_layer(block, self.stage["3"])
        self.stage4 = self._make_layer(block, self.stage["4"])
        self.stage5 = self._make_layer(block, self.stage["5"])
        self.stage6 = self._make_layer(block, self.stage["6"])
        self.stage7 = self._make_layer(block, self.stage["7"])
        self.stage8 = self._make_layer(block, self.stage["8"])

        self.stage9 = nn.Sequential(
            nn.Conv2d(_round_to_multiple_of(320, self.width, self.divisor), 1280, 1, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(1280, num_classes)

    def _make_layer(self, block, stage):
        num_repeats = int(math.ceil(stage["layers"] * self.depth))
        #         inplanes = int(stage["inplanes"] * self.width)
        #         outplanes = int(stage["outplanes"] * self.width)
        inplanes = _round_to_multiple_of(stage["inplanes"], self.width, self.divisor)
        outplanes = _round_to_multiple_of(stage["outplanes"], self.width, self.divisor)

        layers = []
        layers.append(block(inplanes, outplanes, stage["kernel_size"], stride=stage['stride'],
                            expand_ratio=stage["expand_ratio"]))
        for _ in range(1, num_repeats):
            layers.append(block(outplanes, outplanes, stage["kernel_size"], 1, stage["expand_ratio"]))

        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.avg_pooling(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

