# ------------------------------------------------------------------------------
# Copyright (c) NKU
# Licensed under the MIT License.
# Written by Xuanyi Li (xuanyili.edu@gmail.com)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
import time
import datetime


def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):  # conv2d+bn
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation),
        nn.BatchNorm2d(out_channel))


def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):  # conv3d+bn
    return nn.Sequential(
        nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride),
        nn.BatchNorm3d(out_channel))


class BasicBlock(nn.Module):  # conv2d+bn+ReLu+(降采样)+残差
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = convbn(out_channel, out_channel, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        # out = self.conv2(out)
        if self.downsample is not None:  # 如果为none,没有降采样,就只是conv2d+bn+ReLu一次卷积激活
            x = self.downsample(x)
        out = x + out  # residual block
        return out


class FeatureExtraction(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.downsample = nn.ModuleList()
        in_channel = 3
        out_channel = 32
        for _ in range(k):  # k=降采样次数
            self.downsample.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=5,
                    stride=2,
                    padding=2))
            in_channel = out_channel
            out_channel = 32
        self.residual_blocks = nn.ModuleList()
        for _ in range(6):
            self.residual_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1))
        self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb_img):
        output = rgb_img
        for i in range(self.k):
            output = self.downsample[i](output)
        for block in self.residual_blocks:
            output = block(output)
        return self.conv_alone(output)


class EdgeAwareRefinement(nn.Module):  # 最重要步骤，如何利用边缘
    def __init__(self, in_channel):
        super().__init__()
        self.conv2d_feature = nn.Sequential(
            convbn(in_channel, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks = nn.ModuleList()  # astrous天文的？是否和à trous算法有关
        astrous_list = [1, 2, 4, 8, 1, 1]  # 1,2,4,8上采样4次
        for di in astrous_list:  # BasicBlock：conv2d+bn+ReLu
            self.residual_astrous_blocks.append(BasicBlock(32, 32, stride=1, downsample=None, pad=1, dilation=di))
        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, low_disparity, corresponding_rgb):  # 低分辨率一步步向高分辨率refine
        output = torch.unsqueeze(low_disparity, dim=1)
        twice_disparity = F.interpolate(
            output,
            size=corresponding_rgb.size()[-2:],
            mode='bilinear',
            align_corners=False)
        if corresponding_rgb.size()[-1] / low_disparity.size()[-1] >= 1.5:
            twice_disparity *= 8
        output = self.conv2d_feature(
            torch.cat([twice_disparity, corresponding_rgb], dim=1))  # 拼接disp和rgb？rgb应该是提取的边缘吧
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)

        return nn.ReLU(inplace=True)(torch.squeeze(
            twice_disparity + self.conv2d_out(output), dim=1))


class disparityregression(nn.Module):  # 视差回归
    def __init__(self, maxdisp):
        super().__init__()
        self.disp = torch.FloatTensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)  # 感觉是加权求和
        return out


class StereoNet(nn.Module):
    def __init__(self, k=3, r=3, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        self.feature_extraction = FeatureExtraction(k)
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.conv3d_alone = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1)

        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(1):
            self.edge_aware_refinements.append(EdgeAwareRefinement(4))

    def forward(self, left, right):
        # 1、提取特征
        disp = (self.maxdisp + 1) // pow(2, self.k)  # 下采样后,视差范围也对应缩小
        refimg_feature = self.feature_extraction(left)  # reference image feature
        targetimg_feature = self.feature_extraction(right)

        # matching，计算cost volume
        cost = torch.FloatTensor(refimg_feature.size()[0],
                                 refimg_feature.size()[1],
                                 disp,
                                 refimg_feature.size()[2],
                                 refimg_feature.size()[3]).zero_().cuda()  # 猜测是N*F*D*H*W
        for i in range(disp):  # cost volume计算
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[:, :, :, i:] - targetimg_feature[:, :, :, :-i]  # 每个点在不同视差下与对应的右图点相减
            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature
        cost = cost.contiguous()  # 内存连续化,重新分配一块连续的内存

        # cost volume filtering
        for f in self.filter:
            cost = f(cost)
        cost = self.conv3d_alone(cost)
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = disparityregression(disp)(pred)  # 得到粗dispmap

        # hierarchical refinement
        img_pyramid_list = [left]  # 为何加一个列表[]
        pred_pyramid_list = [pred]  # 分别是图像金字塔和预测的dismap金字塔,一层层上采样,img+pred
        pred_pyramid_list.append(self.edge_aware_refinements[0](
            pred_pyramid_list[0], img_pyramid_list[0]))

        for i in range(1):
            pred_pyramid_list[i] = pred_pyramid_list[i] * (
                left.size()[-1] / pred_pyramid_list[i].size()[-1])
            pred_pyramid_list[i] = torch.squeeze(
                F.interpolate(
                    torch.unsqueeze(pred_pyramid_list[i], dim=1),
                    size=left.size()[-2:],
                    mode='bilinear',
                    align_corners=False),
                dim=1)

        return pred_pyramid_list


if __name__ == '__main__':
    model = StereoNet(k=3, r=4).cuda()
    # model.eval()
    # torch.backends.cudnn.benchmark = True
    input = torch.FloatTensor(1, 3, 540, 960).zero_().cuda()  # 测试用

    start = datetime.datetime.now()
    for i in range(100):
        out = model(input, input)
        # shape = [x.size() for x in out]
        # print(shape)
    end = datetime.datetime.now()

    print((end-start).total_seconds())
