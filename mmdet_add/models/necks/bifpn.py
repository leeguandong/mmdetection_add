import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from mmcv.cnn import ConvModule, build_activation_layer
from mmcv.runner import auto_fp16, BaseModule
from mmdet.models import NECKS, FPN
from ..utils import DepthwiseSeparableConvModule, WeightedAdd


@NECKS.register_module()
class BiFPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stack=4,  # repeated bifpn
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.01, eps=1e-3),
                 act_cfg=dict(type='Swish'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(BiFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        assert len(in_channels) == 3, f"Length of input feature maps list should be 3, " \
                                      f"got {len(in_channels_list)}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stack = stack
        self.conv_cfg = conv_cfg
        self.fp16_enabled = False

        self.bfpn = nn.ModuleList()
        for i in range(self.stack):
            self.bfpn.append(
                SingleBiFPN(
                    out_channels=self.out_channels,
                    stack_idx=i,
                    in_channels=self.in_channels,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample_cfg=upsample_cfg))

    def forward(self, x):
        for layer in self.bfpn:
            x = layer(x)
        return x


class SingleBiFPN(nn.Module):
    def __init__(self,
                 out_channels,
                 stack_idx,
                 in_channels,
                 norm_cfg,
                 act_cfg=dict(type='Swish'),
                 upsample_cfg=dict(mode='nearest')):
        super(SingleBiFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.stack_idx = stack_idx
        self.upsample_cfg = upsample_cfg
        self.act_cfg = act_cfg
        self.upsample = partial(F.interpolate, scale_factor=2, **self.upsample_cfg)
        self.downsample = partial(F.max_pool2d, kernel_size=3, stride=2, padding=1)
        self.activate = build_activation_layer(act_cfg)

        if self.stack_idx == 0:
            self.build_on_input_layers()
        self.build_on_lateral_output_layers()

    def build_on_input_layers(self):
        p3_channels, p4_channels, p5_channels = self.in_channels
        in_conv_channels = p3_channels, p4_channels, p4_channels, p5_channels, \
                           p5_channels, p5_channels
        in_conv_names = ('conv_p3_in', 'conv_p4_in_1', 'conv_p4_in_2', 'conv_p5_in_1',
                         'conv_p5_in_2', 'conv_p6_in')
        self.in_convs = nn.ModuleDict()
        for name, in_channels in zip(in_conv_names, in_conv_channels):
            conv_module = ConvModule(
                in_channels,
                self.out_channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
            self.in_convs.update({name: conv_module})

    def build_on_lateral_output_layers(self):
        """ Build top-down and bottom-up paths"""
        # 从上往下，每个节点是两个输入
        wadd_top_down_names = ('wadd_p6_td', 'wadd_p5_td', 'wadd_p4_td', 'wadd_p3_td')
        self.wadd_top_down = nn.ModuleDict()
        for name in wadd_top_down_names:
            wadd_module = WeightedAdd(num_inputs=2)
            self.wadd_top_down.update({name: wadd_module})

        # 从下往上，每个节点是三个输入，包括了从on_input节点来的dense连接
        wadd_bottom_up_names = ('wadd_p4_bu', 'wadd_p5_bu', 'wadd_p6_bu')
        self.wadd_bottom_up = nn.ModuleDict()
        for name in wadd_bottom_up_names:
            wadd_module = WeightedAdd(num_inputs=3)
            self.wadd_bottom_up.update({name: wadd_module})
        # Weighted addition of p7 in bottom up path takes 2 inputs
        self.wadd_bottom_up.update({'wadd_p7_bu': WeightedAdd(num_inputs=2)})

        # Names of depthwise separable convs in top down path
        conv_top_down_names = ('conv_p6_td', 'conv_p5_td', 'conv_p4_td', 'conv_p3_td')
        self.conv_top_down = nn.ModuleDict()
        for name in conv_top_down_names:
            conv_module = DepthwiseSeparableConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                dw_norm_cfg=None,
                dw_act_cfg=None,
                pw_norm_cfg=self.norm_cfg,
                pw_act_cfg=None)
            self.conv_top_down.update({name: conv_module})

        # Names of depthwise separable convs in bottom up path
        conv_bottom_up_names = ('conv_p4_bu', 'conv_p5_bu', 'conv_p6_bu', 'conv_p7_bu')
        self.conv_bottom_up = nn.ModuleDict()
        for name in conv_bottom_up_names:
            conv_module = DepthwiseSeparableConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                dw_norm_cfg=None,
                dw_act_cfg=None,
                pw_norm_cfg=self.norm_cfg,
                pw_act_cfg=None)
            self.conv_bottom_up.update({name: conv_module})

    @auto_fp16()
    def forward(self, inputs):
        if self.stack_idx == 0:
            p3_in, p4_in, p5_in = inputs
            p6_in = self.in_convs['conv_p6_in'](p5_in)
            p6_in = self.downsample(p6_in)
            p7_in = self.downsample(p6_in)

            p4_in_1 = self.in_convs['conv_p4_in_1'](p4_in)
            p5_in_1 = self.in_convs['conv_p5_in_1'](p5_in)

            p4_in_2 = self.in_convs['conv_p4_in_2'](p4_in)
            p5_in_2 = self.in_convs['conv_p5_in_2'](p5_in)

            p3_in = self.in_convs['conv_p3_in'](p3_in)
        else:
            p3_in, p4_in_1, p5_in_1, p6_in, p7_in = inputs
            p4_in_2 = p4_in_1.clone()
            p5_in_2 = p5_in_1.clone()

        # Top down path
        p7_up = self.upsample(p7_in, scale_factor=2, mode="nearest")
        p6_td = self.wadd_top_down['wadd_p6_td']([p6_in, p7_up])
        p6_td = self.activate(p6_td)
        p6_td = self.conv_top_down['conv_p6_td'](p6_td)

        p6_up = self.upsample(p6_td)
        p5_td = self.wadd_top_down['wadd_p5_td']([p5_in_1, p6_up])
        p5_td = self.activate(p5_td)
        p5_td = self.conv_top_down['conv_p5_td'](p5_td)

        p5_up = self.upsample(p5_td)
        p4_td = self.wadd_top_down['wadd_p4_td']([p4_in_1, p5_up])
        p4_td = self.activate(p4_td)
        p4_td = self.conv_top_down['conv_p4_td'](p4_td)

        p4_up = self.upsample(p4_td)
        p3_td = self.wadd_top_down['wadd_p3_td']([p3_in, p4_up])
        p3_td = self.activate(p3_td)
        p3_td = self.conv_top_down['conv_p3_td'](p3_td)

        # Bottom up path
        p3_down = self.downsample(p3_td)
        p4_bu = self.wadd_bottom_up['wadd_p4_bu']([p4_in_2, p4_td, p3_down])
        p4_bu = self.activate(p4_bu)
        p4_bu = self.conv_bottom_up['conv_p4_bu'](p4_bu)

        p4_down = self.downsample(p4_bu)
        p5_bu = self.wadd_bottom_up['wadd_p5_bu']([p5_in_2, p5_td, p4_down])
        p5_bu = self.activate(p5_bu)
        p5_bu = self.conv_bottom_up['conv_p5_bu'](p5_bu)

        p5_down = self.downsample(p5_bu)
        p6_bu = self.wadd_bottom_up['wadd_p6_bu']([p6_in, p6_td, p5_down])
        p6_bu = self.activate(p6_bu)
        p6_bu = self.conv_bottom_up['conv_p6_bu'](p6_bu)

        p6_down = self.downsample(p6_bu)
        p7_bu = self.wadd_bottom_up['wadd_p7_bu']([p7_in, p6_down])
        p7_bu = self.activate(p7_bu)
        p7_bu = self.conv_bottom_up['conv_p7_bu'](p7_bu)

        return p3_td, p4_bu, p5_bu, p6_bu, p7_bu
