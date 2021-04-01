import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

import torch
import torch.nn as nn

from torchvision import models

from argparse import ArgumentParser

from components.resnet import resnet18


class ResNet_Baseline(nn.Module):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        return parser

    def __init__(self, ver_dim, in_channel=3, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32, **kwargs):
        super(ResNet_Baseline, self).__init__()
        resnet18_8s = resnet18(fully_conv=True,
                pretrained=True,
                output_stride=8,
                remove_avg_pool_layer=True)

        self.ver_dim = ver_dim

        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(in_channel+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, ver_dim+1, 1, 1)
        )


    def _normal_initialization(self, layer):
        layer.weight.data_normal(0, 0.01)
        layer.bias.data.zero_()


    def forward(self, image):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(image)

        fm = self.conv8s(torch.cat([xfc,x8s],1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm,x4s],1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm,x2s],1))
        fm = self.up2storaw(fm)

        out = self.convraw(torch.cat([fm,image],1))
        
        offset, confidence = out[:,0:-1,...], out[:,-1,...]

        return offset, confidence



baseline = ResNet_Baseline(8)
frame = torch.zeros(1,3,480,640)
print(baseline(frame)[0].shape)
