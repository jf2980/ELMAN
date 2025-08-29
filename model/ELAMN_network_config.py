import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math as ma
from torchsummary import summary
from thop import profile
import torch.nn.init as init
from torchvision.ops import DeformConv2d
import torchvision.ops
def create_model(args):
    return ELAMN(args)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class DCNConv(nn.Module):
     # Standard convolution
     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
         # ch_in, ch_out, kernel, stride, padding, groups
         super().__init__()
         self.conv1 = nn.Conv2d(c1, c2, 3, 1, 1, groups=g, bias=False)
         deformable_groups = 1
         offset_channels = 18
         self.conv2_offset = nn.Conv2d(c2, deformable_groups * offset_channels, kernel_size=3, padding=1)
         self.conv2 = DeformConv2d(c2, c2, kernel_size=3, padding=1, bias=False)
         #self.conv_ls = nn.Conv2d(c2, c1, 3, 1, 1, groups=g, bias=False)
         # self.conv2 = DeformableConv2d(c2, c2, k, s, autopad(k, p), groups=g, bias=False)
         self.bn1 = nn.BatchNorm2d(c2)
         self.act1 = nn.Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
         self.bn2 = nn.BatchNorm2d(c2)
         self.act2 = nn.Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
     def forward(self, x):
         x = self.act1(self.bn1(self.conv1(x)))
         offset = self.conv2_offset(x)
         x = self.act2(self.bn2(self.conv2(x, offset)))
#         #x=self.conv_ls(x)
         return x
# class DeformableConv2d(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=1,
#                  dilation=1,
#                  groups=3,
#                  bias=False):
#         super(DeformableConv2d, self).__init__()
#
#         assert type(kernel_size) == tuple or type(kernel_size) == int
#
#         kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
#         self.stride = stride if type(stride) == tuple else (stride, stride)
#         self.padding = padding
#         self.dilation = dilation
#
#         self.offset_conv = nn.Conv2d(in_channels,
#                                      2 * kernel_size[0] * kernel_size[1],
#                                      kernel_size=kernel_size,
#                                      stride=stride,
#                                      padding=self.padding,
#                                      dilation=self.dilation,
#                                      groups=groups,
#                                      bias=True)
#
#         nn.init.constant_(self.offset_conv.weight, 0.)
#         nn.init.constant_(self.offset_conv.bias, 0.)
#
#         self.modulator_conv = nn.Conv2d(in_channels,
#                                         1 * kernel_size[0] * kernel_size[1],
#                                         kernel_size=kernel_size,
#                                         stride=stride,
#                                         padding=self.padding,
#                                         dilation=self.dilation,
#                                         groups=groups,
#                                         bias=True)
#
#         nn.init.constant_(self.modulator_conv.weight, 0.)
#         nn.init.constant_(self.modulator_conv.bias, 0.)
#
#         self.regular_conv = nn.Conv2d(in_channels=in_channels,
#                                       out_channels=out_channels,
#                                       kernel_size=kernel_size,
#                                       stride=stride,
#                                       padding=self.padding,
#                                       dilation=self.dilation,
#                                       groups=groups,
#                                       bias=bias)
#
#     def forward(self, x):
#         # h, w = x.shape[2:]
#         # max_offset = max(h, w)/4.
#
#         offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
#         modulator = 2. * torch.sigmoid(self.modulator_conv(x))
#         # op = (n - (k * d - 1) + 2p / s)
#         x = torchvision.ops.deform_conv2d(input=x,
#                                           offset=offset,
#                                           weight=self.regular_conv.weight,
#                                           bias=self.regular_conv.bias,
#                                           padding=self.padding,
#                                           mask=modulator,
#                                           stride=self.stride,
#                                           dilation=self.dilation)
#         return x


# class MeanShift(nn.Conv2d):
#     def __init__(
#         self, rgb_range,
#         rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
#         for p in self.parameters():
#             p.requires_grad = False
# def mean_channels(F):
#     assert(F.dim() == 4)
#     spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
#     return spatial_sum / (F.size(2) * F.size(3))
#
# def stdv_channels(F):
#     assert(F.dim() == 4)
#     F_mean = mean_channels(F)
#     F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
#     return F_variance.pow(0.5)
# class CCALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CCALayer, self).__init__()
#
#         self.contrast = stdv_channels
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#
#     def forward(self, x):
#         y = self.contrast(x) + self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
# def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
#     if not padding and stride == 1:
#         padding = kernel_size // 2
#     return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = 3
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)  ## y为每个通道的权重值
        res = x * y.expand_as(x)
        return res


# class AFFM(nn.Module):  # test_2 dissolution bad
#     def __init__(self, in_channels, SLE=2):
#         super(AFFM, self).__init__()
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         # self.conv1x1_F = nn.Conv2d(in_channels//SLE, in_channels//SLE , kernel_size=1)
#         #         #self.conv1x1_S = nn.Conv2d(in_channels // SLE * 2, in_channels // SLE, kernel_size=1,stride=1)
#         #         #self.conv1x1_T = nn.Conv2d(in_channels // SLE * 2, in_channels // SLE, kernel_size=1, stride=1 )
#         self.conv1x1_L = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         #
#         #         #self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#
#     #         self.SLE = in_channels//SLE
#     #         # for sm in self.modules():
#     #         #     if isinstance(sm, nn.Conv2d):
#     #         #         nn.init.normal_(sm.weight.data, mean=0.0,
#     #         #                         std=ma.sqrt(2 / (sm.out_channels * sm.weight.data[0][0].numel())))
#     #         #         nn.init.zeros_(sm.bias.data)
#     def forward(self, x_list):
#         #         weights = []
#         #         #xa =  torch.cat(x_list,dim=1)
#         #         # x1 = torch.cat([x_list[0],x_list[1]], dim=1)
#         #         # x2 = torch.cat([x_list[0],x_list[2]], dim=1)
#         #         # x3 = torch.cat([x_list[1],x_list[2]], dim=1)
#         #
#
#         U_F = torch.cat(x_list, dim=1)
#         A_F = self.conv1x1_L(U_F)
#         F_F = self.global_avg_pool(A_F)
#         #F_F = self.conv1x1_L(F_F)
#         F_F = self.sigmoid(F_F)
#         fuse = F_F * A_F
#         return fuse
class AFFM(nn.Module):  # test_2 dissolution bad
     def __init__(self, in_channels, SLE=3):
         super(AFFM, self).__init__()
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         # self.conv1x1_F = nn.Conv2d(in_channels//SLE, in_channels//SLE , kernel_size=1)
#         #         #self.conv1x1_S = nn.Conv2d(in_channels // SLE * 2, in_channels // SLE, kernel_size=1,stride=1)
#         #         #self.conv1x1_T = nn.Conv2d(in_channels // SLE * 2, in_channels // SLE, kernel_size=1, stride=1 )
         self.conv1x1_L = nn.Conv2d(in_channels, in_channels, kernel_size=1)
         self.scale_weights = nn.Parameter(torch.ones(SLE))
#         #         #self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#         # self.resi_attn = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d(1),
#         #     nn.Conv2d(in_channels , in_channels// SLE , kernel_size=1),
#         #     nn.ReLU(),
#         #     nn.Conv2d(in_channels// SLE, in_channels , kernel_size=1),
#         #     nn.Sigmoid()
#         # )
#         # self.scale_weights = nn.Parameter(torch.ones(SLE))
#
#     #     self.AFS = nn.Sequential(
#     #         #nn.AdaptiveAvgPool2d(1),
#     #         nn.Conv2d(in_channels, in_channels, kernel_size=1),
#     #         nn.ReLU()
#     # )
#     #         self.SLE = in_channels//SLE
#     #         # for sm in self.modules():
#     #         #     if isinstance(sm, nn.Conv2d):
#     #         #         nn.init.normal_(sm.weight.data, mean=0.0,
#     #         #                         std=ma.sqrt(2 / (sm.out_channels * sm.weight.data[0][0].numel())))
#     #         #         nn.init.zeros_(sm.bias.data)
     def forward(self, x_list):
#         #         weights = []
#         #         #xa =  torch.cat(x_list,dim=1)
#         #         # x1 = torch.cat([x_list[0],x_list[1]], dim=1)
#         #         # x2 = torch.cat([x_list[0],x_list[2]], dim=1)
#         #         # x3 = torch.cat([x_list[1],x_list[2]], dim=1)
#         #
#         # U_F = torch.cat(x_list,dim=1)
#         # A_F=self.conv1x1_L(U_F)
#         # F_F=self.global_avg_pool(A_F)
#         # F_F=self.conv1x1_L(F_F)
#         # F_F=self.sigmoid(F_F)
#         # fuse = F_F*A_F
         y_fused_lis = [w * y for w, y in zip(self.scale_weights, x_list)]
         y_fused = torch.cat(y_fused_lis, dim=1)
         y_fused = self.conv1x1_L(y_fused)
#         # fuse=self.sigmoid(y_fused)
#
#         #         # Pathway 1
#         # RF=self.conv1x1_L(FE)
#
#         # fuse=CF+CR
#         # x2 = conv1 + x2
#
#         # con_cob=torch.cat([conv1,conv2,conv3],dim=1)
#         # fuse=self.conv1x1_L(con_cob)
         return y_fused
# class AFFM(nn.Module):  # test_1 bad
#     def __init__(self, in_channels, SLE=3):
#         super(AFFM, self).__init__()
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1x1_f = nn.Conv2d(in_channels // SLE * 2, in_channels // SLE, kernel_size=1)
#         self.conv1x1_S = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.relu = nn.LeakyReLU()
#         # self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#         self.SLE = in_channels // SLE
#
#     # #         # for sm in self.modules():
#     # #         #     if isinstance(sm, nn.Conv2d):
#     # #         #         nn.init.normal_(sm.weight.data, mean=0.0,
#     # #         #                         std=ma.sqrt(2 / (sm.out_channels * sm.weight.data[0][0].numel())))
#     # #         #         nn.init.zeros_(sm.bias.data)
#     def forward(self, x_list):
#         # #         weights = []
#         #          xa =  torch.cat(x_list,dim=1)
#         x1 = torch.cat([x_list[0], x_list[1]], dim=1)
#         x2 = torch.cat([x_list[0], x_list[2]], dim=1)
#         x3 = torch.cat([x_list[1], x_list[2]], dim=1)
#         # #         #gap = x.mean(dim=(2, 3), keepdim=True)
#         # #         #gap=self.global_avg_pool(x)
#         # #
#         # #         # Pathway 1
#         conv1 = self.relu(self.conv1x1_f(x1))
#         conv2 = self.relu(self.conv1x1_f(x2))
#         conv3 = self.relu(self.conv1x1_f(x3))
#         con_cob = torch.cat([conv1, conv2, conv3], dim=1)
#         fuse = self.conv1x1_S(con_cob)
#         # fused = con_cob + xa
#         return fuse
# class AFFM(nn.Module):
#       def __init__(self, in_channels,SLE=3 ):
#           super(AFFM, self).__init__()
#           self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#           self.conv1x1_f = nn.Conv2d(in_channels, in_channels , kernel_size=1)
#           self.conv1x1_S = nn.Conv2d(in_channels//SLE, in_channels//SLE, kernel_size=1)
#           self.relu = nn.LeakyReLU()
# #         #self.softmax = nn.Softmax(dim=1)
#           self.sigmoid = nn.Sigmoid()
#           self.SLE = in_channels//SLE
# # #         # for sm in self.modules():
# # #         #     if isinstance(sm, nn.Conv2d):
# # #         #         nn.init.normal_(sm.weight.data, mean=0.0,
# # #         #                         std=ma.sqrt(2 / (sm.out_channels * sm.weight.data[0][0].numel())))
# # #         #         nn.init.zeros_(sm.bias.data)
#       def forward(self, x_list):
#          weights = []
# #
#          x = torch.cat(x_list, dim=1)
# #
# #         #gap = x.mean(dim=(2, 3), keepdim=True)
#          gap=self.global_avg_pool(x)
# #
# #         # Pathway 1
#          conv = self.relu(self.conv1x1_f(gap))
#          wei=torch.split(conv, self.SLE ,dim=1)
#          for conv in wei:
#              weight = self.sigmoid(self.conv1x1_S(conv))
#              weights.append(weight)
#          weighted_features = [w * x for w, x in zip(weights, x_list)]
#          fused= torch.cat(weighted_features,dim=1)
#          return fused

# class SplitPointMlp(nn.Module):
#     def __init__(self, dim, mlp_ratio=2):
#         super().__init__()
#         hidden_dim = int(dim//2 * mlp_ratio)
#         self.fc = nn.Sequential(
#             nn.Conv2d(dim//2, hidden_dim, 1, 1, 0),
#             nn.GiLU(inplace=True),
#             nn.Conv2d(hidden_dim, dim//2, 1, 1, 0),
#         )
#
#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=1)
#         x1 = self.fc(x1)
#         x = torch.cat([x1, x2], dim=1)
#         return x


# class GCT(nn.Module):
#     def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
#         super(GCT, self).__init__()
#         self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
#         self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
#         self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
#         self.epsilon = epsilon
#         self.mode = mode
#         self.after_relu = after_relu
#
#     def forward(self, x):
#         if self.mode == 'l2':
#             embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
#             norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
#         elif self.mode == 'l1':
#             _x = torch.abs(x) if not self.after_relu else x
#             embedding = _x.sum((2, 3), keepdim=True) * self.alpha
#             norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
#         else:
#             raise ValueError("Unknown mode!")
#
#         gate = torch.tanh(embedding * norm + self.beta)
#         return x * gate

# class eca_layer(nn.Module):
# #      """Constructs a ECA module.
# # #
# # #     Args:
# # #         channel: Number of channels of the input feature map
# # #         k_size: Adaptive selection of kernel size
# # #     """
# # #
#       def __init__(self,  k_size=3):
#           super(eca_layer, self).__init__()
#           self.avg_pool = nn.AdaptiveAvgPool2d(1)
#           self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#           self.sigmoid = nn.Sigmoid()
# # #
#       def forward(self, x):
# # #         # feature descriptor on the global spatial information
#           y = self.avg_pool(x)
# # #
# # #         # Two different branches of ECA module
#           y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
# # #
# # #         # Multi-scale information fusion
#           y = self.sigmoid(y)
#           y = y.expand_as(x)
#           out = x * y
#           return x * y.expand_as(x)
# class BasicConv(nn.Module):
#     def __init__(
#         self,
#         in_planes,
#         out_planes,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         relu=True,
#         bn=True,
#         bias=False,
#     ):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(
#             in_planes,
#             out_planes,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#         )
#         self.bn = (
#             nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
#             if bn
#             else None
#         )
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# class ZPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
#
#
# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 7
#         self.compress = ZPool()
#         self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
#
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.conv(x_compress)
#         scale = torch.sigmoid(x_out)
#         return x * scale
#
#
# class TripletAttention(nn.Module):
#     def __init__(self, no_spatial=False):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()
#         self.hc = AttentionGate()
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.hw = AttentionGate()
#
#     def forward(self, x):
#         x_perm1 = x.permute(0, 2, 1, 3).contiguous()
#         x_out1 = self.cw(x_perm1)
#         x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
#         x_perm2 = x.permute(0, 3, 2, 1).contiguous()
#         x_out2 = self.hc(x_perm2)
#         x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
#         if not self.no_spatial:
#             x_out = self.hw(x)
#             x_out = 1 / 3 * (x_out + x_out11 + x_out21)
#         else:
#             x_out = 1 / 2 * (x_out11 + x_out21)
#         return x_out

# class CC(nn.Module): #learning bias
#     def __init__(self, channel, reduction=16):
#         super(CC, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_mean = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )
#         self.conv_std = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )
#
#     def forward(self, x):
#
#         # mean
#         ca_mean = self.avg_pool(x)
#         #ca_mean = self.conv_mean(ca_mean)
#
#         # std
#         m_batchsize, C, height, width = x.size()
#         x_dense = x.view(m_batchsize, C, -1)
#         ca_std = torch.std(x_dense, dim=2, keepdim=True)
#         ca_std = ca_std.view(m_batchsize, C, 1, 1)
#         #ca_var = self.conv_std(ca_std)
#
#         # Coefficient of Variation
#         # # cv1 = ca_std / ca_mean
#         # cv = torch.div(ca_std, ca_mean)
#         # ram = self.sigmoid(ca_mean + ca_var)
#
#         cc = (ca_mean + ca_std)/2.0
#         return cc
# class MixShiftBlock(nn.Module): # add the latice block
#     r""" Mix-Shifting Block.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         shift_size (int): Shift size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         drop (float, optional): Dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, dim,mix_distance, shift_size, shift_dist, mix_size, layer_scale_init_value=1e-6,
#                  mlp_ratio=4, drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.InstanceNorm2d):
#         super(MixShiftBlock, self).__init__()
#         self.dim = dim
#
#         self.mlp_ratio = mlp_ratio
#         self.fea_ca1 =CC(dim)
#         self.x_ca1 = CC(dim)
#         self.shift_size = shift_size
#         self.shift_dist = shift_dist
#         self.chunk_size = [i.shape[0] for i in torch.chunk(torch.zeros(dim), self.shift_size)]
#         self.mix_distance = mix_distance
#         #self.kernel_size = [(mix_size[i],mix_distance[i]) for i in range(len(mix_size))]
#         self.kernel_size = [
#             [(ms, ms // 2) for ms in inner_list]  # Each inner list corresponds to a set of kernel sizes
#             for inner_list in mix_size
#         ]
#         self.dwconv_lr = nn.ModuleList(
#             [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for
#              chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size[0])])
#         self.dwconv_lr_S = nn.ModuleList(
#             [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for
#              chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size[1])])
#         self.dwconv_td = nn.ModuleList(
#             [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for
#              chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size[0])])
#         self.dwconv_td_S = nn.ModuleList(
#             [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for
#              chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size[1])])
#
#         self.norm = norm_layer(dim)
#         self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.layer = len(mix_size)
#     def forward(self, x):
#         input = x
#
#         # for i in range(self.layer):
#         # split groups
#         xs = torch.chunk(x, self.shift_size, 1)
#
#         # shift with pre-defined relative distance
#         x_shift_lr = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, self.shift_dist)]
#         x_shift_td = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, self.shift_dist)]
#         x_shift_lr_S = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, self.shift_dist)]
#         x_shift_td_S = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, self.shift_dist)]
#         x_ca1 = [self.fea_ca1(x) for x in x_shift_lr_S]
#         # regional mixing
#         for i in range(self.shift_size):
#             x_shift_lr[i] = self.dwconv_lr[i](x_shift_lr[i])
#             x_shift_td[i] = self.dwconv_td[i](x_shift_td[i])
#         fea_ca1_lr = [self.fea_ca1(x_shift_lr[i]) for i in range(len(x_shift_lr))]
#         fea_ca1_td = [self.fea_ca1(x_shift_td[i]) for i in range(len(x_shift_td))]
#         for i in range(len(x_shift_lr_S)):
#             x_shift_lr_S[i] = x_shift_lr_S[i] * fea_ca1_lr[i] + x_shift_lr[i]
#         for i in range(len(x_shift_lr_S)):
#             x_shift_td_S[i] = x_shift_td_S[i] * fea_ca1_td[i] + x_shift_td[i]
#         for i in range(self.shift_size):
#             x_shift_lr_S[i] = self.dwconv_lr_S[i](x_shift_lr[i])
#             x_shift_td_S[i] = self.dwconv_td_S[i](x_shift_td[i])
#         for i in range(len(x_shift_lr)):
#             x_shift_lr[i] = x_ca1[i] * x_shift_lr[i] + x_shift_lr_S[i]
#         for i in range(len(x_shift_td)):
#             x_shift_td[i] = x_ca1[i] * x_shift_td[i] + x_shift_td_S[i]
#         x_lr = torch.cat(x_shift_lr, 1)
#         x_td = torch.cat(x_shift_td, 1)
#         x_lr_s = torch.cat(x_shift_lr_S, 1)
#         x_td_s = torch.cat(x_shift_td_S, 1)
#         # add more layer
#         x = x_lr + x_td + x_lr_s + x_td_s
#         x = self.norm(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#         x = self.drop_path(x)
#
#         return x



class ShiftConv2d0(nn.Module): #shift trianguler
     def __init__(self, inp_channels, out_channels):
         super().__init__()
         self.inp_channels = inp_channels
         self.out_channels = out_channels
         self.n_div = 5
         g = inp_channels // self.n_div
#
         conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
         mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
         mask[:, 0 * g:1 * g, 1, 2] = 1.0 ## left
         mask[:, 0 * g:1 * g, 0, 2] = 1.0 ##lefttdown
         mask[:, 1 * g:2 * g, 1, 0] = 1.0 ## right
         mask[:, 1 * g:2 * g, 2, 0] = 1.0 ## rightdown
         mask[:, 2 * g:3 * g, 2, 1] = 1.0
         mask[:, 3 * g:4 * g, 0, 1] = 1.0
         mask[:, 4 * g:, 1, 1] = 1.0
         self.w = conv3x3.weight
         self.b = conv3x3.bias
         self.m = mask
#
     def forward(self, x):
         y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
         return y





class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        #self.weight[0 * g:1 * g, 0,0 ,0 ] = 1.0  ## up
        # #self.weight[0 * g:1 * g, 0, 0, 0] = 1.0 ##upleft
        # #self.weight[0 * g:1 * g, 0, 1, 1] = 1.0  ##leftup
        #self.weight[1 * g:2 * g, 0, 2, 2] = 1.0  ## right
        # #self.weight[1 * g:2 * g, 0, 0, 2] = 1.0
        # #self.weight[1 * g:2 * g, 0, 2, 0] = 1.0  ## rightdown
        #self.weight[2 * g:3 * g, 0, 1, 0] = 1.0  ## rightdown
        # #self.weight[2 * g:3 * g, 0, 0, 0] = 1.0
        # #self.weight[2 * g:3 * g, 0, 2, 2] = 1.0  ##upleft
        #
        #self.weight[3 * g:4 * g, 0, 2, 1] = 1.0  ## down
        # #self.weight[3 * g:4 * g, 0, 2, 2] = 1.0
        # #self.weight[3 * g:4 * g, 0, 2, 0] = 1.0  ## downright
        #self.weight[4 * g:5 * g, 0, 2, 0] = 1.0  ## middown
        #self.weight[5 * g:, 0, 1, 1] = 1.0  ## identity
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        # self.weight[2 * g:3 * g, 0, 0, 0] = 1.0  ##right up
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)
        self.relu =nn.ReLU(inplace=True)
    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y =self.conv1x1(y)

        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed',exp_ratio=4,):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class LFE(nn.Module): # Change to the pixel convolution
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu',shared_depth=1):
        super().__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type
        self.sigmoid = nn.Sigmoid()
        self.exp_ratio =exp_ratio
        self.conv0 = ShiftConv2d1(inp_channels, out_channels*self.exp_ratio )
        self.conv1 = ShiftConv2d1(out_channels*self.exp_ratio , out_channels)
        # if self.act_type == 'linear':
        #     self.act = None
        if self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        # elif self.act_type == 'selu':
        #     self.act = nn.SELU()
        else:
            raise ValueError('unsupport type of activation')
        self.batchnorm =nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)

        return y


class GMMSA(nn.Module):
    def __init__(self,hed, channels,shifts=4 ,window_sizes=[4, 8, 12], calc_attn=True,num_global_tokens=4,shared_depth=1,mlp_ratio=2,drop_path=0.):
        super().__init__()
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn
        self.hidden_size = channels//2
        self.hed = hed
        self.num_global_tokens = num_global_tokens
        #self.chan_conv= nn.Conv2d(self.channels, self.channels, kernel_size=1)
        #self.GCT = GCT(self.channels)
        self.fusion= AFFM(in_channels=channels)

        if self.calc_attn:
            self.split_chns = [channels * 2 // 3, channels * 2 // 3, channels * 2 // 3]
            self.project_inp = nn.Sequential(

                nn.Conv2d(self.channels, self.channels * 2, kernel_size=1),
                nn.GroupNorm(num_groups=2,num_channels=self.channels * 2),

                #
            )
            self.project_out = nn.Sequential(

                nn.Conv2d(self.channels, self.channels, kernel_size=1),

            )

        else:
            self.split_chns = [channels  // 3, channels  // 3, channels  // 3]
            self.project_inp = nn.Sequential(

                nn.Conv2d(self.channels, self.channels , kernel_size=1),
                nn.GroupNorm(num_groups=1, num_channels=self.channels ),

                #
            )
        self.project_out = nn.Sequential(
            # GCT(self.channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=1),

        )
        #self.mlp=SplitPointMlp(channels,mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_scores = nn.Parameter(torch.zeros(8 , 8))
        nn.init.normal_(self.token_scores, std=0.02)
        # for sm in self.modules():
        #     if isinstance(sm, nn.Conv2d):
        #         nn.init.normal_(sm.weight.data, mean=0.0,
        #                         std=ma.sqrt(2 / (sm.out_channels * sm.weight.data[0][0].numel())))
        #         nn.init.zeros_(sm.bias.data)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape

        x = self.project_inp(x)


        ##change position
        #weights = torch.softmax(self.merge_weights, dim=0)
        xs = torch.split(x, self.split_chns, dim=1)
        #xs = [PA(nf=int(c*2/3))(xs[x_i]) for x_i in range(len(xs))]
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):  # xs = 4 = [2,4,8,16]
                wsize = self.window_sizes[idx]
                hed  = self.hed[idx]
                if self.shifts > 0:
                    #x_ = torch.roll(x_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                    x_ = torch.roll(x_, shifts=(wsize // 2, -wsize // 2), dims=(2, 3))
                    #x_ = torch.roll(x_, shifts=(wsize // 2,0 ), dims=(2, 3))
                    #x_ = torch.roll(x_, shifts=(wsize // 2, -wsize // 2), dims=(2, 3))
                else:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, wsize // 2), dims=(2, 3))
                    #x_ = torch.roll(x_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                #x_=self.linear_o(x_)
                #x_norm = torch.nn.functional.normalize(x_, dim=(2, 3))
                q, v = rearrange(
                    x_, 'b (qv c he) (h dh) (w dw) -> qv  (b h w ) he (dh dw) c ',
                    qv=2, dh=wsize, dw=wsize, he= hed
                )
                q_l = q * (hed ** -0.5)

                k_l = q_l.transpose(-2, -1)

                attn_local = torch.matmul(q_l, k_l)
                attn_local = attn_local.softmax(dim=-1)
                out_local = torch.matmul(attn_local, v)
                out_attn = attn_local.clone()
                #
                # #glbal part
                scores = self.token_scores  # shape [head, num_tokens]
                topk = torch.topk(scores, k=self.num_global_tokens, dim=-1)
                global_idx = torch.zeros(wsize * wsize, dtype=torch.bool, device=x.device)
                global_idx[topk.indices[0]] = True
                q_global = q_l[:, :, global_idx, :]
                attn_global = torch.matmul(q_global, k_l)
                attn_global = attn_global.softmax(dim=-1)
                out_attn[:, :, global_idx, :] = attn_global
                out_global = torch.matmul(attn_global, v)
                out_win = out_local.clone()
                out_win[:, :, global_idx, :] = out_global
                #y_ = y_ / 2.0
                # attn_local = torch.matmul(q_l, k_l)
                # attn_local = attn_local.softmax(dim=-1)
                # out_local = torch.matmul(attn_local, v)
                # out_attn = attn_local.clone()
                #
                # global_idx = torch.zeros(wsize * wsize, dtype=torch.bool, device=x.device)
                # global_idx[:self.num_global_tokens] = True
                # q_global = q_l[:, :, global_idx, :]
                # attn_global = torch.matmul(q_global, k_l)
                # attn_global = attn_global.softmax(dim=-1)
                # out_attn[:, :, global_idx, :] = attn_global
                # out_global = torch.matmul(attn_global, v)
                # out_win = out_local.clone()
                # out_win[:, :, global_idx, :] = out_global
                y_ = rearrange(
                    out_win, '  (b h w ) he (dh dw) c   -> b (c he) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize, he=hed
                )
                if self.shifts > 0:
                    #y_ = torch.roll(y_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                    #y_ = torch.roll(y_, shifts=(-wsize // 2,0  ), dims=(2, 3))
                    #y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2 ), dims=(2, 3))
                    y_ = torch.roll(y_, shifts=(-wsize // 2, wsize // 2), dims=(2, 3))
                else:
                    y_ = torch.roll(y_, shifts=(wsize // 2, -wsize // 2), dims=(2, 3))
                    #y_ = torch.roll(y_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                ys.append(y_ )
                atns.append(out_attn)
            #y = torch.cat(ys, dim=1)
            #y = self.project_out(y)
            y=self.fusion(ys)
            #y = self.drop_path(self.mlp(y)) + y
            return y, atns
        else:
            for idx, x_ in enumerate(xs):  # xs = 4 = [2,4,8,16]
                wsize = self.window_sizes[idx]
                hed = self.hed[idx]
                if self.shifts > 0:
                    #x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                    x_ = torch.roll(x_, shifts=(wsize // 2, -wsize // 2), dims=(2, 3))
                    #x_ = torch.roll(x_, shifts=(wsize // 2,0 ), dims=(2, 3))
                else:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, wsize // 2), dims=(2, 3))
                    #x_ = torch.roll(x_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                       #x_ = torch.roll(x_, shifts=(0, -wsize // 2), dims=(2, 3))
                #x_=self.linear_t(x_)
                #x_norm = torch.nn.functional.normalize(x_, dim=(2, 3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c he) (h dh) (w dw) ->  (b h w ) he (dh dw) c ',
                    dh=wsize, dw=wsize ,he= hed
                )
                y_ = (atn @ v)

                y_ = rearrange(
                    y_, ' (b h w ) he (dh dw) c-> b (c he) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize, he= hed
                )
                if self.shifts > 0:
                    #y_ = torch.roll(y_, shifts=(-wsize // 2,0 ), dims=(2, 3))
                    #y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                    y_ = torch.roll(y_, shifts=(-wsize // 2, wsize // 2), dims=(2, 3))
                    #y_ = torch.roll(y_, shifts=(wsize // 2, 0), dims=(2, 3))
                else:
                    y_ = torch.roll(y_, shifts=(wsize // 2, -wsize // 2), dims=(2, 3))
                    #y_ = torch.roll(y_, shifts=(0, wsize // 2), dims=(2, 3))
                    #y_ = torch.roll(y_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                ys.append(y_)
            #y = torch.cat(ys, dim=1)
            #y=self.project_out(y)
            y = self.fusion(ys)
            return y, prev_atns


class ELMAB(nn.Module):
    def __init__(self,hed,inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=[4, 8, 12], shared_depth=1,mlp_ratio=4.,act_layer=nn.GELU,drop_path=0.):
        super().__init__()
        self.hed = hed
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth
        #self.GCT =GCT(self.inp_channels)
        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe['lfe_0'] = LFE(inp_channels=self.inp_channels, out_channels=self.out_channels, exp_ratio=self.exp_ratio,shared_depth=shared_depth)
        modules_gmsa['gmsa_0'] = GMMSA(channels=inp_channels,hed=self.hed ,shifts= self.shifts, window_sizes=window_sizes, calc_attn=True,shared_depth=shared_depth)
        self.eca_layer = eca_layer(k_size=3)
        #self.modules_lfe = nn.ModuleList()
        #self.modules_gmsa = nn.ModuleList()

        # self.modules_lfe.append(
        #     LFE(inp_channels=self.inp_channels, out_channels=self.out_channels, exp_ratio=self.exp_ratio,
        #         shared_depth=shared_depth))
        # self.modules_gmsa.append(
        #     GMSA(channels=inp_channels, shifts=self.shifts, window_sizes=window_sizes, calc_attn=True,
        #          shared_depth=shared_depth))
        for i in range(shared_depth):
            modules_lfe['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                       exp_ratio=exp_ratio)
            modules_gmsa['gmsa_{}'.format(i + 1)] = GMMSA(channels=inp_channels, shifts= self.shifts,
                                                          hed=hed,window_sizes=window_sizes, calc_attn=False)
        # for i in range(shared_depth):
        #     self.modules_lfe.append(LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio))
        #     self.modules_gmsa.append(
        #         GMSA(channels=inp_channels, shifts=self.shifts, window_sizes=window_sizes, calc_attn=False))
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.ECA =eca_layer()#effeicent channel block
        self.mlp = nn.Sequential(nn.Conv2d(inp_channels,inp_channels,1),nn.ReLU())
    def forward(self, x):
        atn = None

        #x = self.GCT(x)
        for i in range(1 + self.shared_depth):
             if i == 0:  # Only calculate attention for the 1st module
                 #EC=self.eca_layer(x)

                 x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                 y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, None)
                 #x = y + x

                 x = self.mlp(self.drop_path(y)) + x
                 #
             else:
                 #EC = self.eca_layer(x)

                 x = self.modules_lfe['lfe_{}'.format(i)](x) + x  # Use the previous output
                 y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, atn)
                 #x = y + x
                 x = self.mlp(self.drop_path(y))+ x
        #x = self.GCT(x)
        return x
        # atn = None
        # for i in range(1 + self.shared_depth):
        #     if i == 0:  # Only calculate attention for the 1st module
        #         x = self.modules_lfe[i](x) + x
        #         y, atn = self.modules_gmsa[i](x, None)
        #         x = y + x
        #     else:
        #         x = self.modules_lfe[i](x) + x  # Use the previous output
        #         y, atn = self.modules_gmsa[i](x, atn)
        #         x = y + x


#class FrequencyFFC(nn.Module):
#     def __init__(self, num_channels):
#         super(FrequencyFFC, self).__init__()
#         self.num_channels = num_channels
#         #
#         self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
#         self.relu1 = nn.LeakyReLU(inplace=True)
#         #
#
#
#     #
#     def forward(self, x):
#         x_real = torch.real(x)
#         x_real = x_real.float()
#         #
#         out = self.conv1(x_real)
#         out = self.relu1(out)
#         #
#
#         #
#         return out


# class FourierUnit(nn.Module):
#     def __init__(self, embed_dim, fft_norm='ortho'):
#         #         # bn_layer not used
#         super(FourierUnit, self).__init__()
#         self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
#         self.relu = nn.LeakyReLU(negative_slope=0.2)
#         #
#         self.fft_norm = fft_norm
#
#
#     def forward(self, x):
#         batch = x.shape[0]
#         #
#         r_size = x.size()
#         #
#         #         # (batch, c, h, w/2+1, 2)
#         fft_dim = (-2, -1)
#         ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
#         ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
#         ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
#         #
#         ffted = ffted.view((batch, -1,) + ffted.size()[3:])
#         #         #a = (batch, -1,) + ffted.size()[3:]
#         #         #b =ffted.view((batch, -1,)
#         ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
#         ffted = self.relu(ffted)
#         #
#         ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
#                                                                        2).contiguous()  # (batch,c, t, h, w/2+1, 2)
#         ffted = torch.complex(ffted[..., 0], ffted[..., 1])
#
#         ifft_shape_slice = x.shape[-2:]
#         output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
#         #         #if torch.isnan(output).any():
#         #             #output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
#         return output
#
#
#
# class SpectralTransform(nn.Module):
#     def __init__(self, embed_dim, last_conv=False):
#         # bn_layer not used
#         super(SpectralTransform, self).__init__()
#         self.last_conv = last_conv
#         #
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
#             #
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         )
#         self.fu = FourierUnit(embed_dim // 2)
#         #
#         self.conv2 = nn.Sequential(nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0),
#                                    #                                    #nn.BatchNorm2d(embed_dim, eps=1e-4),
#                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
#         #                                    #nn.ELU(inplace=True))
#         #        #self.la_conv=torch.nn.Conv2d(embed_dim*2,embed_dim,1, 1, 0)
#         if self.last_conv:
#             self.last_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         x = self.conv1(x)
#         output = self.fu(x)
#         #         #output =self.ECA(x)
#         output = self.conv2(output + x)
#         #
#         #         #output = torch.cat((x,output),dim=1)
#         #         #output=self.la_conv( output)
#         if self.last_conv:
#             output = self.last_conv(output)
#         return output


## Residual Block (RB)
class REB(nn.Module):
    def __init__(self, embed_dim, red=2):
        super(REB, self).__init__()
        self.body = nn.Sequential(
            DCNConv(embed_dim, embed_dim // red),
            nn.LeakyReLU(0.2, inplace=True),
            DCNConv(embed_dim // red,embed_dim ),
        )
        #self.DformCon= DCNConv( embed_dim, embed_dim,)
    def forward(self, x):
        out = self.body(x)
        #out = self.ECA(out)
        return out + x


# class SFB(nn.Module):
#      def __init__(self, embed_dim, red=1):
#          super(SFB, self).__init__()
#          self.S = ResDB(embed_dim, red)
#          self.F = FourierUnit(embed_dim)
#          self.fusion = nn.Sequential(nn.Conv2d(embed_dim*2 , embed_dim, 1, 1, 0),
#                                      nn.LeakyReLU(0.2, inplace=True),
#                                      #nn.BatchNorm2d(embed_dim,eps=1e-2,track_running_stats=False),
#                                      )
# #
#      def forward(self, x):
#          s = self.S(x)
#          f = self.F(x)
#          out = torch.cat([s, f], dim=1)
#          out = self.fusion(out)
#          return out


# Perform forward pass
#output = elan_model(sample_input)
class ELAMN(nn.Module):
    def __init__(self, config,drop_path = 0.):
        super().__init__()

        self.config = config
        self.hed = config['hed']
        self.scale = config['scale']
        self.colors = config['colors']  # Assuming RGB
        self.r_expand = config['r_expand']  # Assuming some value
        self.c_elan = config['c_elan']  # Assuming some value
        self.m_elan = config['m_elan']  # Assuming some value
        self.n_share = config['n_share']  # Assuming some value
        self.window_sizes = config['window_sizes']  #
        self.sub_mean = MeanShift(config['rgb_range'])
        self.add_mean = MeanShift(config['rgb_range'], sign=1)
        #self.sub_mean = MeanShift(1.)
        #self.add_mean = MeanShift(1., sign=1)
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        #self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        #self.ELAB=ELAB(inp_channels=self.c_elan, out_channels=self.c_elan,window_sizes= self.window_sizes)
        self.head = nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)
        num_layers = self.m_elan // (1 + self.n_share)
        #init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='')
        # configs = [
        #     (0 if (i + 1) % 2 == 1 else 1)  # shift based on index
        #     for i in range(self.m_elan // (1 + self.n_share))
        # ]
        #
        # m_body = [ ELAB(
        #             self.hed,self.c_elan, self.c_elan, self.r_expand, shifts,
        #             self.window_sizes, self.n_share
        #         )for shifts in configs
        # ]
        m_body = [
             ELMAB(
                 self.hed,
                 self.c_elan,
                 self.c_elan,
                 self.r_expand,
                 i % 2,  # This will be 0 for even indices and 1 for odd indices
                 self.window_sizes,
                 self.n_share
             )
             for i in range(num_layers)
         ]

        self.tail = nn.Sequential(
             nn.Conv2d(self.c_elan, self.colors * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
             nn.PixelShuffle(self.scale)
         )

        self.conv_after_body= nn.Conv2d(self.c_elan, self.c_elan, kernel_size=3, stride=1, padding=1)

        # self.conv_before_upsam = nn.Sequential(nn.Conv2d(self.c_elan, self.c_elan, kernel_size=3, stride=1, padding=1),
        #                                      nn.LeakyReLU(inplace=True), )
        self.conv_last = nn.Conv2d(self.colors, self.colors, kernel_size=3, stride=1, padding=1)
        self.REB =REB(self.c_elan)
        #self.body = nn.ModuleList(body)

        #self.head = nn.Sequential(*m_head)
        # self.head = m_head
        self.body = nn.Sequential(*m_body)
        #self.tail = nn.Sequential(*m_tail)
        # self.MixShiftBlock_MLP = MixShiftBlock(mix_size=mix_size, shift_dist=shift_dist, mix_distance=mix_distance,
        #                                        dim=self.c_elan, shift_size=5)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()



    def forward(self, x):
         H, W = x.shape[2:]
         #x=torch.nn.init.uniform_(x, a=0.0, b=1.0)
         #x = self.sub_mean(x)
         x = self.check_image_size(x)
         x = self.sub_mean(x)
         x = self.head(x)

         res = self.body(x)

         res = self.REB(res)
         res=self.conv_after_body(res)

         res=res+x
         x=self.tail(res)
         #x=self.conv_last(x)
         #x = x / 1. + self.mean
         x=self.add_mean(x)
         return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    pass
# Dummy class ELAB and SFB
# class ELAB(nn.Module):
#     pass
#
# class SFB(nn.Module):
#     pass

# Dummy input
# input_image = torch.randn(1, 3, 64, 64)  # Assuming input size is 64x64
#
# # Create an instance of the ELAN model
# model = ELAN(scale=2, colors=3, r_expand=1, c_elan=60, m_elan=10, n_share=2, window_sizes=[8, 16, 32], )
#
# # Forward pass
# output_image = model(input_image)
#
# # Print the shape of the output
# flops, params = profile(model, inputs=(input_image,))
# print('parameters:', params / 1e6, 'flops:', flops / 1e9)
# # Print the output shape (batch size, num_classes)
#
# print("Output shape:", output_image.shape)
