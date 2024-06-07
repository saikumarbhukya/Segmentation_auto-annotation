import torch
import torch.nn as nn
#from torchsummary import summary


# Convolutions for Basic Block conserving the dimensions of the input
def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 bottleneck convolution with padding """
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 5x5 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 7x7 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, groups=groups, bias=False, dilation=dilation)


# Convolutions for downsampling and upsampling
def downConv(planes, stride=2, groups=1, dilation=1):
    """
        3x3 downsampling convolution
        as a substitute for Max Pool
    """
    return nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)


def upConv(planes, kernel_size=2, dilation=1, stride=2):
    """
        upsampling transposed convolution
    """
    return nn.ConvTranspose2d(in_channels=planes, out_channels=planes,
                              kernel_size=kernel_size, dilation=dilation, stride=stride)


class BasicBlock(nn.Module):
    """
        Convolution + ReLU +
        Convolution + ReLU
    """

    def __init__(self, inplanes, outplanes, groups=1, downsample=False, resconn=True, batchnorm=True):
        super(BasicBlock, self).__init__()
        self.resconn = resconn
        self.downsample = downsample
        self.batchnorm = batchnorm
        self.bottleneck = conv1x1(inplanes, outplanes)

        self.conv1 = conv7x7(inplanes, outplanes)
        self.conv2 = conv5x5(inplanes, outplanes)
        self.conv3 = conv3x3(inplanes, outplanes)
        self.batchnorm_func = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

        if self.downsample is True:
            self.down_layer = downConv(outplanes)

    def forward(self, x):
        identity = x
        out = self.conv1(x) + self.conv2(x) + self.conv3(x)
        if self.batchnorm:
            out = self.batchnorm_func(out)
        out = self.relu(out)

        if self.resconn is True:
            identity = self.bottleneck(identity)
            out += identity

        if self.downsample is True:
            out = self.down_layer(out)

        out = self.relu(out)
        return out


class Unet(nn.Module):
    """
        Modularized U-Net
    """

    def __init__(self):
        super(Unet, self).__init__()
        self.nbc = 6
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Encoding
        self.layer1 = BasicBlock(3, self.nbc, groups=1)
        self.layer2 = BasicBlock(self.nbc, 2 * self.nbc)
        self.layer3 = BasicBlock(2 * self.nbc, 4 * self.nbc)
        self.layer4 = BasicBlock(4 * self.nbc, 8 * self.nbc)
        self.layer5 = BasicBlock(8 * self.nbc, 16 * self.nbc)

        # Decoding
        self.upsamp5 = upConv(16 * self.nbc)
        self.layer5_ = BasicBlock((16 + 8) * self.nbc, 8 * self.nbc)

        self.upsamp4 = upConv(8 * self.nbc)
        self.layer4_ = BasicBlock((8 + 4) * self.nbc, 4 * self.nbc)

        self.upsamp3 = upConv(4 * self.nbc)
        self.layer3_ = BasicBlock((4 + 2) * self.nbc, 2 * self.nbc)

        self.upsamp2 = upConv(2 * self.nbc)
        self.layer2_ = BasicBlock((2 + 1) * self.nbc, self.nbc)

        self.layer1_0 = conv3x3(self.nbc, 6)
        self.layer1_1 = conv3x3(6, 3)
        self.layer1_2 = conv1x1(3, 1)

    def forward(self, x):
        # Encoding
        x1 = self.layer1(x)
        x1_pool = self.pool(x1)

        x2 = self.layer2(x1_pool)
        x2_pool = self.pool(x2)

        x3 = self.layer3(x2_pool)
        x3_pool = self.pool(x3)

        x4 = self.layer4(x3_pool)
        x4_pool = self.pool(x4)

        x5 = self.layer5(x4_pool)

        x5_up = torch.cat([self.upsamp5(x5), x4], dim=1)
        x5_ = self.layer5_(x5_up)

        x4_up = torch.cat([self.upsamp4(x5_), x3], dim=1)
        x4_ = self.layer4_(x4_up)

        x3_up = torch.cat([self.upsamp3(x4_), x2], dim=1)
        x3_ = self.layer3_(x3_up)

        x2_up = torch.cat([self.upsamp2(x3_), x1], dim=1)
        x2_ = self.layer2_(x2_up)

        x1_ = self.layer1_2(self.layer1_1(self.layer1_0(x2_)))
        out = self.sigmoid(x1_)

        return out


if __name__ == '__main__':
    print("Model architecture: ")
    model = Unet().cuda()
    # summary(model, (1, 512, 512))