import torch
from torch import nn
import torch.nn.functional as F
from .modules import ResGCN_Module


class ResGCN_Input_Branch(nn.Module):
    def __init__(self, structure, block, num_channel, A, **kwargs):
        super(ResGCN_Input_Branch, self).__init__()

        self.register_buffer('A', A)
        # 构造分支框架
        module_list = [ResGCN_Module(num_channel, 64, 'Basic', A, initial=True, **kwargs)]
        module_list += [ResGCN_Module(64, 64, 'Basic', A, initial=True, **kwargs) for _ in range(structure[0] - 1)]
        module_list += [ResGCN_Module(64, 32, block, A, **kwargs)]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    # 每个分支的处理方式
    def forward(self, x):
        x = self.bn(x)
        for layer in self.layers:
            x = layer(x, self.A)

        return x


t_feature = []

# class GeMHPP(nn.Module):
    # # def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
    # def __init__(self, bin_num=[1], p=6.5, eps=1.0e-6):
    #     super(GeMHPP, self).__init__()
    #     self.bin_num = bin_num
    #     self.p = nn.Parameter(
    #         torch.ones(1)*p)
    #     self.eps = eps
    #
    # def gem(self, ipts):
    #     return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)
    #
    # def forward(self, x):
    #     """
    #         x  : [n, c, h, w]
    #         ret: [n, c, p]
    #     """
    #     n, c = x.size()[:2]
    #     features = []
    #     for b in self.bin_num:
    #         z = x.view(n, c, b, -1)
    #         z = self.gem(z).squeeze(-1)
    #         features.append(z)
    #     return torch.cat(features, -1)
class ResGCN(nn.Module):
    def __init__(self, module, structure, block, num_input, num_channel, num_class, A, A2, **kwargs):
        super(ResGCN, self).__init__()

        self.register_buffer('A', A)
        self.register_buffer('A2', A2)
        # self.HPP = GeMHPP()

        # input branches
        self.input_branches = nn.ModuleList([
            ResGCN_Input_Branch(structure, block, num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [module(32 * num_input, 128, block, A, stride=1, is_main_stream=True, **kwargs)]
        module_list += [module(128, 128, block, A, is_main_stream=True, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, block, A, stride=1, is_main_stream=True, **kwargs)]
        module_list += [module(256, 256, block, A, is_main_stream=True, **kwargs) for _ in range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)
        # --------------------------------------------------------------------------------------------------------------
        # # # main stream_2
        module_list2 = [module(32 * num_input, 128, block, A2, stride=1, is_main_stream=True, **kwargs)]
        module_list2 += [module(128, 128, block, A2, is_main_stream=True, **kwargs) for _ in range(structure[2] - 1)]
        module_list2 += [module(128, 256, block, A2, stride=1, is_main_stream=True, **kwargs)]
        module_list2 += [module(256, 256, block, A2, is_main_stream=True, **kwargs) for _ in range(structure[3] - 1)]
        self.main_stream2 = nn.ModuleList(module_list2)
        # --------------------------------------------------------------------------------------------------------------

        # 平均池化
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.fcn = nn.Linear(256, num_class)

        # self.conv_1 = nn.Conv2d(32, 32, 1)
        # self.conv_1 = nn.Linear(1, 1)
        self.conv_11 = nn.Conv2d(32, 32, 1)
        self.conv_21 = nn.Conv2d(32, 32, 1)
        self.conv_31 = nn.Conv2d(32, 32, 1)
        self.conv_41 = nn.Conv2d(32, 32, 1)
        self.conv_51 = nn.Conv2d(32, 32, 1)

        # self.conv_1 = nn.Conv2d(128, 128, 1)
        # self.conv_2 = nn.Conv2d(256, 256, 1)
        # self.fcn_1 = nn.Linear(32,32)

        self.tran_metric = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
             ], dtype=torch.float)

        # self.finalconv = 0

        # init parameters，初始化参数
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):
        global t_feature
        # N, I, C, T, V = x.size()

        # input branches
        x_cat = []
        # 会进入branch 23行那个网络架构，包含s_basic和t_basic
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:, i, :, :, :]))

        x = torch.cat(x_cat, dim=1)  # (16,64,600,32)
        x1 = x.clone()

        # main stream，用来聚合拼接信息
        # -------------------------------------------------------------------------------------------------------------
        """
        高阶邻域聚合，把身体划分成5个部分
        input: x2,A2
        """
        b = torch.sum(x[:, :, :, 0:4], dim=3, keepdim=True)  # x改变，里面的值不会发生改变,其他维度不变，只把最后的32->5
        c = torch.sum(x[:, :, :, 26:32], dim=3, keepdim=True)
        mid = b + c
        l_hand = x[:, :, :, 4:11]
        r_hand = x[:, :, :, 11:18]
        l_leg = x[:, :, :, 18:22]
        r_leg = x[:, :, :, 22:26]

        #         a = mid.max(dim=3, keepdim=True)[0]
        #         b = l_hand.max(dim=3, keepdim=True)[0]
        #         c = r_hand.max(dim=3, keepdim=True)[0]
        #         d = l_leg.max(dim=3, keepdim=True)[0]
        #         e = r_leg.max(dim=3, keepdim=True)[0]

        #         a = mid.median(dim=3, keepdim=True)[0]
        #         b = l_hand.median(dim=3, keepdim=True)[0]
        #         c = r_hand.median(dim=3, keepdim=True)[0]
        #         d = l_leg.median(dim=3, keepdim=True)[0]
        #         e = r_leg.median(dim=3, keepdim=True)[0]

        a = mid.sum(dim=3, keepdim=True) / 10
        b = l_hand.sum(dim=3, keepdim=True) / 7
        c = r_hand.sum(dim=3, keepdim=True) / 7
        d = l_leg.sum(dim=3, keepdim=True) / 4
        e = r_leg.sum(dim=3, keepdim=True) / 4
        a = self.conv_11(a)
        b = self.conv_21(b)
        c = self.conv_31(c)
        d = self.conv_41(d)
        e = self.conv_51(e)

        #
        x2 = torch.cat((a, b, c, d, e), -1)  # 求和或者平均或者中位数之后也是用cat拼接
        #
        # #         x2 = self.fcn_pool(x2)
        #
        for layer2 in self.main_stream2:
            x2 = layer2(x2, self.A2)
            t_feature.append(x2.detach().clone())
        # --------------------------------------------------------------------------------------------------------------
        """
        低阶邻域聚合，融合全局特征进行特征提取
        input: x1,  A,  t_feature,  tran_metric
        """
        i = 0
        for layer in self.main_stream:
            x1 = layer(x1, self.A)
            x1 = x1 + t_feature[i] @ (self.tran_metric.cuda())

            # cat + 点卷积  + 全连接
            # x1 = torch.cat((x1, t_feature[i] @ (self.tran_metric.cuda())), -2)
            if i <= 1:
                x1 = self.conv_1(x1)
            else:
                x1 = self.conv_2(x1)
            # x1 = self.fcn_1(x1)
            i += 1
        t_feature = []
        # --------------------------------------------------------------------------------------------------------------
        # output，平均池化
        # x = self.global_pooling(x)
        # x = self.global_pooling(x1) + self.global_pooling(x2)\
        # self.finalconv = x1.detach()

        x = self.global_pooling(x1)
        # x = self.HPP(x1)

        x = self.fcn(x.squeeze())

        # # L2 normalization
        x = F.normalize(x, dim=1, p=2)

        return x


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    for m in modules:
        if isinstance(m, ResGCN_Module):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)
