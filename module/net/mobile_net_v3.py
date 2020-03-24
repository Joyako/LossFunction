import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class HSwish(nn.Module):
    """The hard version of swish"""
    def forward(self, x):
        return x * F.relu6(x + 3., inplace=True) / 6.


class HSigmoid(nn.Module):
    """The hard version of sigmoid"""
    def forward(self, x):

        return F.relu6(x + 3., inplace=True)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNHS(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNHS, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            HSwish()
        )


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, SE=None):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.nl = HSwish()
        self.se = SE

    def forward(self, x):
        x = self.nl(self.bn(self.conv(x)))
        if self.se is not None:
            x = self.se(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, exp_size, stride, NL, SE=False):
        super(BasicBlock, self).__init__()
        if SE:
            self.se = SELayer(planes)
        else:
            self.se = None

        if NL == 'RE':
            self.nl = nn.ReLU()
        else:
            self.nl = HSwish()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(inplanes, exp_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_size)
        self.nl1 = self.nl
        self.conv2 = nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=exp_size, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.nl2 = self.nl
        self.conv3 = nn.Conv2d(exp_size, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.skip_connect = None

        if stride == 1 and inplanes == planes:
            self.skip_connect = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.nl1(self.bn1(self.conv1(x)))
        out = self.nl2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.se is not None:
            out = self.se(out)

        if self.skip_connect is not None:
            out = out + self.skip_connect(x)

        return out


class MobileNetSmall(nn.Module):
    def __init__(self, num_bins=66, width_mult=1.0, round_nearest=8):
        super(MobileNetSmall, self).__init__()
        bneck = BasicBlock
        input_channel = 16

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNHS(3, input_channel, 3, 2)]
        out_channels = [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96]
        kernel_size = [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5]
        stride = [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1]
        exp_size = [16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576]
        NL = ['RE', 'RE', 'RE', 'HS', 'HS', 'HS', 'HS', 'HS', 'HS', 'HS', 'HS']
        SE = [True, False, False, True, True, True, True, True, True, True, True, True]

        for out, k, s, e, nl, se in zip(out_channels, kernel_size, stride, exp_size, NL, SE):
            output_channel = _make_divisible(out * width_mult, round_nearest)
            features.append(bneck(input_channel, output_channel, k, e, s, nl, se))
            input_channel = output_channel

        features.append(ConvBlock(output_channel, 576, 1, 1, SELayer(576)))
        self.classifier = nn.Sequential(
            nn.Linear(576, 1280),
            HSwish(),
        )

        self.fc1 = nn.Linear(1280, num_bins)
        self.fc2 = nn.Linear(1280, num_bins)
        self.fc3 = nn.Linear(1280, num_bins)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)

        yaw = self.fc1(x)
        pitch = self.fc2(x)
        roll = self.fc3(x)

        return yaw, pitch, roll


class MobileNetLarge(nn.Module):
    def __init__(self, num_bins=66, width_mult=1.0, round_nearest=8):
        super(MobileNetLarge, self).__init__()
        bneck = BasicBlock
        input_channel = 16

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNHS(3, input_channel, 3, 2)]
        out_channels = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160]
        kernel_size = [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
        stride = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]
        exp_size = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960]
        NL = ['RE', 'RE', 'RE', 'RE', 'RE', 'RE', 'HS', 'HS', 'HS', 'HS', 'HS', 'HS', 'HS', 'HS', 'HS']
        SE = [False, False, False, True, True, True, False, False, False, False, True, True, True, True, True]

        for out, k, s, e, nl, se in zip(out_channels, kernel_size, stride, exp_size, NL, SE):
            output_channel = _make_divisible(out * width_mult, round_nearest)
            features.append(bneck(input_channel, output_channel, k, e, s, nl, se))
            input_channel = output_channel

        features.append(ConvBlock(output_channel, 960, 1, 1, None))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            HSwish(),
        )

        self.fc1 = nn.Linear(1280, num_bins)
        self.fc2 = nn.Linear(1280, num_bins)
        self.fc3 = nn.Linear(1280, num_bins)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)

        yaw = self.fc1(x)
        pitch = self.fc2(x)
        roll = self.fc3(x)

        return yaw, pitch, roll


