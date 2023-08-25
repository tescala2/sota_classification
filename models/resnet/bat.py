import torch
import torch.nn as nn
import torch.utils.model_zoo as mz

from torch.nn import functional as F
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional, Tuple

# https://github.com/BA-Transform/BAT-Image-Classification

__all__ = ['NLNet', 'resnet18_bat', 'resnet34_bat', 'resnet50_bat', 'resnet101_bat', 'resnet152_bat']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_nonlocal_block(block_type):
    block_dict = {'nl': NonLocal, 'bat': BATBlock}
    if block_type in block_dict:
        return block_dict[block_type]
    else:
        raise ValueError("UNKOWN NONLOCAL BLOCK TYPE:", block_type)


class SEBlock(nn.Module):
    def __init__(self, planes, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(planes, planes // r),
            nn.ReLU(inplace=True),
            nn.Linear(planes // r, planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = self.squeeze(x)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(x.size(0), x.size(1), 1, 1)

        return x * excitation.expand_as(x)


class NonLocalModule(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(NonLocalModule, self).__init__()

    def init_modules(self):
        for name, m in self.named_modules():
            if len(m._modules) > 0:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif len(list(m.parameters())) > 0:
                raise ValueError("UNKOWN NONLOCAL LAYER TYPE:", name, m)


class NonLocal(NonLocalModule):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """

    def __init__(self, inplanes, use_scale=True, **kwargs):
        planes = inplanes // 2
        self.use_scale = use_scale

        super(NonLocal, self).__init__(inplanes)
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1,
                           stride=1, bias=True)
        self.bn = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            att = att.div(c ** 0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, h, w)

        x = self.z(x)
        x = self.bn(x) + residual

        return x


class BATransform(nn.Module):
    def __init__(self, in_channels, s, k):
        super(BATransform, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, k, 1),
                                   nn.BatchNorm2d(k),
                                   nn.ReLU(inplace=True))
        self.conv_p = nn.Conv2d(k, s * s * k, [s, 1])
        self.conv_q = nn.Conv2d(k, s * s * k, [1, s])
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))
        self.s = s
        self.k = k
        self.in_channels = in_channels

    def extra_repr(self):
        return 'BATransform({in_channels}, s={s}, k={k})'.format(**self.__dict__)

    def resize_mat(self, x, t):
        n, c, s, s1 = x.shape
        assert s == s1
        if t <= 1:
            return x
        x = x.view(n * c, -1, 1, 1)
        x = x * torch.eye(t, t, dtype=x.dtype, device=x.device)
        x = x.view(n * c, s, s, t, t)
        x = torch.cat(torch.split(x, 1, dim=1), dim=3)
        x = torch.cat(torch.split(x, 1, dim=2), dim=4)
        x = x.view(n, c, s * t, s * t)
        return x

    def forward(self, x):
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.s, 1))
        cp = F.adaptive_max_pool2d(out, (1, self.s))
        p = self.conv_p(rp).view(x.size(0), self.k, self.s, self.s)
        q = self.conv_q(cp).view(x.size(0), self.k, self.s, self.s)
        p = torch.sigmoid(p)
        q = torch.sigmoid(q)
        p = p / p.sum(dim=3, keepdim=True)
        q = q / q.sum(dim=2, keepdim=True)
        p = p.view(x.size(0), self.k, 1, self.s, self.s).expand(x.size(
            0), self.k, x.size(1) // self.k, self.s, self.s).contiguous()
        p = p.view(x.size(0), x.size(1), self.s, self.s)
        q = q.view(x.size(0), self.k, 1, self.s, self.s).expand(x.size(
            0), self.k, x.size(1) // self.k, self.s, self.s).contiguous()
        q = q.view(x.size(0), x.size(1), self.s, self.s)
        p = self.resize_mat(p, x.size(2) // self.s)
        q = self.resize_mat(q, x.size(2) // self.s)
        y = p.matmul(x)
        y = y.matmul(q)

        y = self.conv2(y)
        return y


class BATBlock(NonLocalModule):
    def __init__(self, in_channels, r=2, s=4, k=4, dropout=0.2, **kwargs):
        super().__init__(in_channels)

        inter_channels = in_channels // r
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(inplace=True))
        self.batransform = BATransform(inter_channels, s, k)
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        xl = self.conv1(x)
        y = self.batransform(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x

    def init_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            use_se: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se_block = SEBlock(planes * self.expansion) if use_se else nn.Identity()
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

        out = self.se_block(out)
        out += residual
        out = self.relu(out)

        return out


class NLNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            nltype: str = 'nl',  # 'nl' or 'bat'
            nl_mod: List[int] = [2, 2, 1000],
            k: int = 4,
            transpose: bool = False,
            nlsize: Tuple[int] = (7, 7, 7),
            dropout: float = 0.2,
            use_se: bool = False,
            bb_fc: bool = True
    ) -> None:
        super(NLNet, self).__init__()
        self.nltype = nltype
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.transpose = transpose
        self.dropout = dropout
        self.dilation = 1
        self.use_se = use_se
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       nl_mod=nl_mod[0], s=nlsize[0], k=k)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       nl_mod=nl_mod[1], s=nlsize[1], k=k)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       nl_mod=nl_mod[2], s=nlsize[2], k=k)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bb = bb_fc
        if self.bb:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, NonLocalModule):
                m.init_modules()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, nl_mod=1000, s=7, k=4):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        for i in range(blocks):
            if i == 0:
                layers.append((str(i), block(self.inplanes, planes, stride, downsample, self.groups,
                                             self.base_width, previous_dilation, norm_layer, use_se=self.use_se)))
                self.inplanes = planes * block.expansion
            else:
                layers.append((str(i), block(self.inplanes, planes, groups=self.groups,
                                             base_width=self.base_width, dilation=self.dilation,
                                             norm_layer=norm_layer, use_se=self.use_se)))
            if i % nl_mod == nl_mod - 1:
                layers.append(
                    ('nl{}'.format(i),
                     get_nonlocal_block(self.nltype)(self.inplanes, s=s, k=k, transpose=self.transpose,
                                                     dropout=self.dropout)))
                # print('add {} after block {} with {} planes.'.format(
                #     self.nltype, i, self.inplanes))

        return nn.Sequential(OrderedDict(layers))

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
        if self.bb:
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18_bat(pretrained=False, **kwargs) -> NLNet:
    """Constructs a NLNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NLNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = mz.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnet34_bat(pretrained=False, **kwargs) -> NLNet:
    """Constructs a NLNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NLNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = mz.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnet50_bat(pretrained=False, **kwargs) -> NLNet:
    """Constructs a NLNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NLNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = mz.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnet101_bat(pretrained=False, **kwargs) -> NLNet:
    """Constructs a NLNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NLNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = mz.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnet152_bat(pretrained=False, **kwargs) -> NLNet:
    """Constructs a NLNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NLNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = mz.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


if __name__ == '__main__':
    num_classes = 8

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    x = torch.randn(64, 3, 224, 224).to(device)

    model = resnet152_bat(pretrained=False, num_classes=num_classes).to(device)

    print(model)
    print(model(x).size())
