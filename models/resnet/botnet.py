import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

from torch.utils import model_zoo

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

__all__ = ['ResNet', 'resnet18_bot', 'resnet34_bot', 'resnet50_bot', 'resnet101_bot',
           'resnet152_bot', 'resnext50_32x4d_bot', 'resnext101_32x8d_bot',
           'wide_resnet50_2_bot', 'wide_resnet101_2_bot']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BoTNet_Layer(nn.Module):
    def __init__(self, in_channels, num_heads=4, image_size=224, inference=False):
        super(BoTNet_Layer, self).__init__()
        self.num_heads = num_heads
        self.dk = self.dv = in_channels

        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dv % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"

        self.kqv_conv = nn.Conv2d(in_channels, 2 * self.dk + self.dv, kernel_size=1).to(device)

        # Positional encodings
        self.rel_encoding_h = nn.Parameter(
            torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        self.rel_encoding_w = nn.Parameter(
            torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        dkh = self.dk // self.num_heads
        dvh = self.dv // self.num_heads
        flatten_hw = lambda x, depth: torch.reshape(x, (batch_size, self.num_heads, height * width, depth))

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        k, q, v = torch.split(kqv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (dkh ** -0.5)

        # After splitting, shape is [batch_size, num_heads, height, width, dkh or dvh]
        k = self.split_heads_2d(k, self.num_heads)
        q = self.split_heads_2d(q, self.num_heads)
        v = self.split_heads_2d(v, self.num_heads)

        # [batch_size, num_heads, height*width, height*width]
        qk = torch.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh).transpose(2, 3))

        qr_h, qr_w = self.relative_logits(q)
        qk += qr_h
        qk += qr_w

        weights = F.softmax(qk, dim=-1)

        if self.inference:
            self.weights = nn.Parameter(weights)

        attn_out = torch.matmul(weights, flatten_hw(v, dvh))
        attn_out = torch.reshape(attn_out, (batch_size, self.num_heads, self.dv // self.num_heads, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        return attn_out

    # Split channels into multiple heads.
    def split_heads_2d(self, inputs, num_heads):
        batch_size, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads, height, width, depth // num_heads)
        split_inputs = torch.reshape(inputs, ret_shape)
        return split_inputs

    # Combine heads (inverse of split heads 2d).
    def combine_heads_2d(self, inputs):
        batch_size, num_heads, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads * depth, height, width)
        return torch.reshape(inputs, ret_shape)

    # Compute relative logits for both dimensions.
    def relative_logits(self, q):
        _, num_heads, height, width, dkh = q.size()
        rel_logits_w = self.relative_logits_1d(q, self.rel_encoding_w, height, width, num_heads, [0, 1, 2, 4, 3, 5])
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.rel_encoding_h, width, height, num_heads,
                                               [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w

    # Compute relative logits along one dimenion.
    def relative_logits_1d(self, q, rel_k, height, width, num_heads, transpose_mask):
        rel_logits = torch.einsum('bhxyd,md->bxym', q, rel_k)
        # Collapse height and heads
        rel_logits = torch.reshape(rel_logits, (-1, height, width, 2 * width - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it
        rel_logits = torch.reshape(rel_logits, (-1, height, width, width))
        # Tile for each head
        rel_logits = torch.unsqueeze(rel_logits, dim=1)
        rel_logits = rel_logits.repeat((1, num_heads, 1, 1, 1))
        # Tile height / width times
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, height, 1, 1))
        # Reshape for adding to the logits.
        rel_logits = rel_logits.permute(transpose_mask)
        rel_logits = torch.reshape(rel_logits, (-1, num_heads, height * width, height * width))
        return rel_logits

    # Converts tensor from relative to absolute indexing.
    def rel_to_abs(self, x):
        # [batch_size, num_heads*height, L, 2L−1]
        batch_size, num_heads_height, L, _ = x.size()
        # Pad to shift from relative to absolute indexing.
        col_pad = torch.zeros((batch_size, num_heads_height, L, 1)).to(device)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (batch_size, num_heads_height, L * 2 * L))
        flat_pad = torch.zeros((batch_size, num_heads_height, L - 1)).to(device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        # Reshape and slice out the padded elements.
        final_x = torch.reshape(flat_x_padded, (batch_size, num_heads_height, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


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
            image_size: int = 224,
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
            self.conv2 = BoTNet_Layer(num_heads=num_heads, in_channels=width, image_size=image_size,
                                      inference=inference)
        self.bn2 = norm_layer(planes)
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
            image_size: int = 224,
            inference: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride)
        self.bn1 = norm_layer(width)
        if not attention:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = BoTNet_Layer(num_heads=num_heads, in_channels=width, image_size=image_size,
                                      inference=inference)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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
            attention: List[bool] = [False, False, False, False],
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
                                       image_size=image_size // 4, inference=inference)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], attention=attention[1],
                                       num_heads=num_heads, image_size=image_size // 8, inference=inference)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], attention=attention[2],
                                       num_heads=num_heads, image_size=image_size // 16, inference=inference)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], attention=attention[3],
                                       num_heads=num_heads, image_size=image_size // 32, inference=inference)
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
                    dilate: bool = False, attention: bool = False, num_heads: int = 8, image_size: int = 224,
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
                            image_size=image_size, inference=inference))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attention=attention, num_heads=num_heads, image_size=image_size,
                                inference=inference))

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



def resnet18_bot(pretrained: bool = False, **kwargs: Any) -> ResNet:
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


def resnet34_bot(pretrained: bool = False, **kwargs: Any) -> ResNet:
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


def resnet50_bot(pretrained: bool = False, **kwargs: Any) -> ResNet:
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


def resnet101_bot(pretrained: bool = False, **kwargs: Any) -> ResNet:
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


def resnet152_bot(pretrained: bool = False, **kwargs: Any) -> ResNet:
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


def resnext50_32x4d_bot(pretrained: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnext50_32x4d'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def resnext101_32x8d_bot(pretrained: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnext101_32x8d'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def wide_resnet50_2_bot(pretrained: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['wide_resnet50_2'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


def wide_resnet101_2_bot(pretrained: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['wide_resnet101_2'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict, strict=False)
    return model


if __name__ == '__main__':
    attention = [True, True, True, True]
    num_classes = 50

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    x = torch.randn(2, 3, 224, 224).to(device)

    model = resnet50_bot(pretrained=True, attention=attention, num_classes=num_classes).to(device)

    print(model)
    print(model(x).size())
