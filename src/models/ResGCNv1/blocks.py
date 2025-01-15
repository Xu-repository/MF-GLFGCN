import torch
import torch.nn as nn


class Spatial_Bottleneck_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, reduction=8, is_main_stream=False,
                 **kwargs):
        super(Spatial_Bottleneck_Block, self).__init__()

        inter_channels = out_channels // 6

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        # -----------------------------------------------------------------------------------
        self.is_main_stream = is_main_stream
        #         if self.is_main_stream:
        #             max_graph_distance, self.max_graph_distance1 = max_graph_distance[0], max_graph_distance[1]
        # -----------------------------------------------------------------------------------

        self.conv_down = nn.Conv2d(in_channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        self.conv = SpatialGraphConv(inter_channels, inter_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(inter_channels)

        self.conv_up = nn.Conv2d(inter_channels, out_channels, 1)
        self.bn_up = nn.BatchNorm2d(out_channels)

        self.relu = nn.LeakyReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block)

        if self.is_main_stream:   # 这个影响不大
            x = self.pool(x)

        return x


class Temporal_Bottleneck_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=8, **kwargs):
        super(Temporal_Bottleneck_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // 6

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride, 1)),
                nn.BatchNorm2d(channels),
            )

        self.conv_down = nn.Conv2d(channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        self.conv = nn.Conv2d(inter_channels, inter_channels, (temporal_window_size, 1), (stride, 1), padding)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, channels, 1)
        self.bn_up = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block + res_module)

        return x


class Spatial_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, is_main_stream=False, **kwargs):
        super(Spatial_Basic_Block, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x


class Temporal_Basic_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, **kwargs):
        super(Temporal_Basic_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride, 1)),
                nn.BatchNorm2d(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size, 1), (stride, 1), padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)

        x = self.relu(x + res_block + res_module)

        return x


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels * self.s_kernel_size, 1)

    def forward(self, x, A):
        # numbers in same class have same weight
        x = self.gcn(x)
        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()

        return x
