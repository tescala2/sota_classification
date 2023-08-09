import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

__all__ = ['ResNet', 'resnet18_saaa', 'resnet34_saaa', 'resnet50_saaa', 'resnet101_saaa', 'resnet152_saaa']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Axial_Layer(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=56, stride=1, height_dim=True, inference=False):
        super(Axial_Layer, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads

        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False).to(device)
        self.kqv_bn = nn.BatchNorm1d(self.depth * 2).to(device)
        self.logits_bn = nn.BatchNorm2d(num_heads * 3).to(device)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size).to(device)
        query_index = torch.arange(kernel_size).to(device)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size * kernel_size))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width

        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)

        # Compute q, k, v
        kqv = self.kqv_conv(x).to(device)
        kqv = self.kqv_bn(kqv)  # apply batch normalization on k, q, v
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height),
                              [self.dh // 2, self.dh // 2, self.dh], dim=2)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2,
                                                                                               self.kernel_size,
                                                                                               self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits)  # apply batch normalization on qk, qr, kr
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)

        weights = F.softmax(logits, dim=3)

        if self.inference:
            self.weights = nn.Parameter(weights)

        attn = torch.matmul(weights, v.transpose(2, 3)).transpose(2, 3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)

        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)

        return output


class BasicBlock(nn.Module):
    expansion: int = 1

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
            attention: bool = False,
            num_heads: int = 8,
            kernel_size: int = 56,
            inference: bool = False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        width = int(planes * (base_width / 64.)) * groups
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if not attention:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.height_block = Axial_Layer(in_channels=width, num_heads=num_heads, kernel_size=kernel_size,
                                            inference=inference)
            self.width_block = Axial_Layer(in_channels=width, num_heads=num_heads, kernel_size=kernel_size, stride=1,
                                           height_dim=False, inference=inference)
        #    self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

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
            attention: bool = False,
            num_heads: int = 8,
            kernel_size: int = 56,
            inference: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.attention = attention
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride)
        self.bn1 = norm_layer(width)
        if not attention:
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(width)
        else:
            self.height_block = Axial_Layer(in_channels=width, num_heads=num_heads, kernel_size=kernel_size,
                                            inference=inference)
            self.width_block = Axial_Layer(in_channels=width, num_heads=num_heads, kernel_size=kernel_size, stride=1,
                                           height_dim=False, inference=inference)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.attention:
            out = self.conv2(out)
            out = self.bn2(out)
        else:
            out = self.height_block(out)
            out = self.width_block(out)
        #       out = self.bn2(out)

        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
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
            attention: List[bool] = [True, True, True, True],
            num_heads: int = 8,
            image_size: int = 224,
            inference: bool = False
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
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
        self.layer1 = self._make_layer(block, 64, layers[0], attention=attention[0], num_heads=num_heads,
                                       kernel_size=image_size // 4, inference=inference)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       attention=attention[1], num_heads=num_heads, kernel_size=image_size // 8,
                                       inference=inference)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       attention=attention[2], num_heads=num_heads, kernel_size=image_size // 16,
                                       inference=inference)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       attention=attention[3], num_heads=num_heads, kernel_size=image_size // 32,
                                       inference=inference)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if num_classes == 1000:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Sequential(nn.Linear(512 * block.expansion, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1,
                    dilate: bool = False, attention: bool = False, num_heads: int = 8, kernel_size: int = 56,
                    inference: bool = False) -> nn.Sequential:
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, attention, num_heads=num_heads,
                            kernel_size=kernel_size, inference=inference))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attention=attention, num_heads=num_heads,
                                kernel_size=kernel_size, inference=inference))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18_saaa(pretrained: bool = False, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnet34_saaa(pretrained: bool = False, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnet50_saaa(pretrained: bool = False, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnet101_saaa(pretrained: bool = False, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnet152_saaa(pretrained: bool = False, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


if __name__ == '__main__':
    attention = [True, True, True, True]
    num_classes = 50

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    x = torch.randn(2, 3, 224, 224).to(device)

    model = resnet50_saaa(attention=attention, num_classes=num_classes).to(device)

    print(model)
    print(model(x).size())
