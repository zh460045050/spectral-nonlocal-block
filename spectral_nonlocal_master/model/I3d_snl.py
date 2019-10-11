import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

from models.snl_block import st_SNLStage, st_gSNLStage
__all__ = [
    'Backbone', 'Nonlocal', 'NLStage', 'SNL', 'gSNL', 'CGNL', 'A2Net'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class NonlocalNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400,
                 nl_type=None, nl_nums=None, stage_num=None, out_num=None, div=2):
        self.inplanes = 64
        super(NonlocalNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)


        if not nl_nums:
            self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2,
                                           nl_type=nl_type, nl_nums=nl_nums, stage_num = stage_num, out_num=out_num, div = div)

        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)


        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if nl_nums == 1:
            for name, m in self._modules['layer3'][-2].named_modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def _addNonlocal(self, in_planes, sub_planes, nl_type='nl', stage_num=None, out_num=None, use_scale=False, groups=8, order=3, relu=False, aff_kernel='dot'):

            if nl_type == 'snl':
                return st_SNLStage(
                    in_planes, sub_planes,
                    use_scale=False, stage_num=stage_num, out_num = out_num,
                    relu=relu, aff_kernel=self.aff_kernel)
            elif nl_type == 'gsnl':
                return st_gSNLStage(
                    in_planes, sub_planes,
                    use_scale=False, stage_num=stage_num, relu=relu, aff_kernel=self.aff_kernel)
            else:
                raise KeyError("Unsupported nonlocal type: {}".format(nl_type))

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, nl_type=None, nl_nums=None, stage_num=None, out_num=None, div=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        sub_planes = int(self.inplanes / div)
        for i in range(1, blocks):
            if nl_nums == 1 and (i == 5 and blocks == 6) or (i == 22 and blocks == 23) or (i == 35 and blocks == 36):
                layers.append(self._addNonlocal(self.inplanes,sub_planes, nl_type, stage_num, out_num))
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):


    if ft_begin_index == 0:
        return model.parameters()
    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')
    #print(ft_module_names)
    ft_module_names.append('layer3.5')

    parameters = []
    for k, v in model.named_parameters():
        print(k)
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters




def Backbone(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type=None, nl_nums=0, stage_num=0, out_num=0, **kwargs)
    return model

def Nonlocal(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type='originalstage', nl_nums=1, stage_num=1, out_num=0, **kwargs)
    return model

def NLStage(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type='nlstage', nl_nums=1, stage_num=1, out_num=0, **kwargs)
    return model

def CGNL(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type='cgnlstage', nl_nums=1, stage_num=1, out_num=0, **kwargs)
    return model

def GNL_3(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type='mygnl', nl_nums=1, stage_num=1, out_num=3, **kwargs)
    return model


def GNL_2(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type='mygnl', nl_nums=1, stage_num=1, out_num=2, **kwargs)
    return model

def gSNL(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type='channelgnlstage', nl_nums=1, stage_num=1, out_num=3, **kwargs)
    return model


def SNL(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type='channelgnlstage', nl_nums=1, stage_num=1, out_num=2, **kwargs)
    return model

def A2Net(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = NonlocalNet(Bottleneck, [3, 4, 6, 3], nl_type='a2stage', nl_nums=1, stage_num=1, out_num=0, **kwargs)
    return model
