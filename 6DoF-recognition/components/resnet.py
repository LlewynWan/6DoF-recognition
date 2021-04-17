import torch
import torch.nn as nn

import math
import numpy as np

import torch.utils.model_zoo as model_zoo
from typing import Type, Any, Callable, Union, List, Optional


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv1x1(in_channel: int, out_channel: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channel: int, out_channel: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    kernel_size = np.asarray((3,3))
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    full_padding = (upsampled_kernel_size - 1) // 2
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
            stride=stride, padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channel: int, num_channel: int,
            stride: int = 1, dilation: int = 1,
            downsample: Optional[nn.Module] = None) -> None:
        
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channel, num_channel, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_channel, num_channel, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(num_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channel: int, num_channel: int,
            stride: int = 1, dilation: int = 1,
            downsample: Optional[nn.Module] = None):

        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channel, num_channel)
        self.bn1 = nn.BatchNorm2d(num_channel)

        self.conv2 = conv3x3(num_channel, num_channel, stride=stride, dilation=dilation)
        
        self.bn2 = nn.BatchNorm2d(num_channel)
        self.conv3 = conv1x1(num_channel, num_channel * 4)
        self.bn3 = nn.BatchNorm2d(num_channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
            block,
            layers,
            in_channel=3,
            num_classes=1000,
            fully_conv=False,
            remove_avg_pool_layer=False,
            output_stride=32):

        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1
        
        self.remove_avg_pool_layer = remove_avg_pool_layer

        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
         
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:
                
                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride
                
            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=self.current_dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.current_dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x2s = self.relu(x)
        x = self.maxpool(x2s)

        x4s = self.layer1(x)
        x8s = self.layer2(x4s)
        x16s = self.layer3(x8s)
        x32s = self.layer4(x16s)
        x=x32s
        
        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)
        
        if not self.fully_conv:
            x = x.view(x.size(0), -1)
            
        xfc = self.fc(x)

        return x2s,x4s,x8s,x16s,x32s,xfc


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='./models'))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet34'],model_dir=proj_cfg.MODEL_DIR))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet50'],model_dir=proj_cfg.MODEL_DIR))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet101'],model_dir=proj_cfg.MODEL_DIR))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet152'],model_dir=proj_cfg.MODEL_DIR))
    return model
