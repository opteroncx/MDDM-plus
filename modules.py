# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
from MPNCOV.python import MPNCOV
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

class RK3(nn.Module):
    def __init__(self, n_feats=64, kernel_size=3,bias=True, act=nn.PReLU(1, 0.25), res_scale=1):

        super(RK3, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)
        self.scale5 = nn.Parameter(torch.FloatTensor([1/6]), requires_grad=True)

    def forward(self, x):
        
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        yn_1 = k1*self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2*k2
        yn_2 = yn_2 + k1*self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2*self.scale4 + k1
        yn_3 = yn_3*self.scale5
        out = yn_3 + yn
        return out

class Space_attention(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Space_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale
        # downscale = scale + 4

        self.K = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)
        #self.bn = nn.BatchNorm2d(output_size)
        if kernel_size == 1:
            self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)


    def forward(self, x):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        # Q = F.interpolate(Q, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            Q = self.pool(Q)
        else:
            Q = Q
        V = self.V(x)
        # V = F.interpolate(V, scale_factor=1 / self.scale, mode='bicubic')
        if self.stride > 1:
            V = self.pool(V)
        else:
            V = V
        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)
        # if self.type == 'softmax':
        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.matmul(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)

        vector = torch.matmul(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        W = self.local_weight(O)
        output = x + W
        #output = self.bn(output)
        return output

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)

        return self.act(out)


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def din(content_feat,encode_feat,eps=None):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    encode_mean, encode_std = calc_mean_std(encode_feat)
    if eps==None:
        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)        
    else:
        normalized_feat = (content_feat - content_mean.expand(
            size)) / (content_std.expand(size)+eps)
    return normalized_feat * encode_std.expand(size) + encode_mean.expand(size)


class Down2(nn.Module):
    def __init__(self,c_in,c_out):
        super(Down2, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()            

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.down(out)
        LR_2x = self.convt_R1(out)
        return LR_2x


class ScaleLayer(nn.Module):
   def __init__(self, init_value=1.0):
       super(ScaleLayer,self).__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, x):
       return x * self.scale

class Sobel_dw(nn.Module):
    def __init__(self, in_channels):
        super(Sobel_dw, self).__init__()
        self.conv_op_x = nn.Conv2d(3, 3, 3,stride=1, padding=1, bias=False,groups=3)
        self.conv_op_y = nn.Conv2d(3, 3, 3,stride=1, padding=1, bias=False,groups=3)
        sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]],dtype='float32')
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],dtype='float32')
        sobel_kernel_x = torch.from_numpy(sobel_kernel_x).unsqueeze(0)
        sobel_kernel_y = torch.from_numpy(sobel_kernel_y).unsqueeze(0)
        sobel_kernel_x = sobel_kernel_x.unsqueeze(0).expand(in_channels, -1, -1, -1)
        sobel_kernel_y = sobel_kernel_y.unsqueeze(0).expand(in_channels, -1, -1, -1)
        self.conv_op_x.weight.data = sobel_kernel_x
        self.conv_op_y.weight.data = sobel_kernel_y
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    def forward(self, x):
        edge_Y_x = self.conv_op_x(x)
        edge_Y_y = self.conv_op_y(x)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)
        return edge_Y

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        # self.device = device
        self.conv_op_x = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_y = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        # self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        # self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    def forward(self, x):
        edge_Y_x = self.conv_op_x(x)
        edge_Y_y = self.conv_op_y(x)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)
        return edge_Y

class AdvancedSobel(nn.Module):
    def __init__(self):
        super(AdvancedSobel, self).__init__()
        self.conv_op_x = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_y = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_a45 = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_a135 = nn.Conv2d(3, 1, 3,stride=1, padding=1, bias=False)
        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_a45 = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, -0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, -0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, -0]]], dtype='float32')
        sobel_kernel_a135 = np.array([[[2, 1, 0], [1, 0, -1], [0, -1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0, -1, -2]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))
        sobel_kernel_a45 = sobel_kernel_a45.reshape((1, 3, 3, 3))
        sobel_kernel_a135 = sobel_kernel_a135.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y)
        self.conv_op_a45.weight.data = torch.from_numpy(sobel_kernel_a45)
        self.conv_op_a135.weight.data = torch.from_numpy(sobel_kernel_a135)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False
        self.conv_op_a45.weight.requires_grad = False
        self.conv_op_a135.weight.requires_grad = False

    def forward(self, x):
        edge_Y_x = self.conv_op_x(x)
        edge_Y_y = self.conv_op_y(x)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)
        return edge_Y

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError
# ----------------------------------------------------------------
class L1_Wavelet_Loss_RW(nn.Module):
    def __init__(self):
        super(L1_Wavelet_Loss_RW, self).__init__()
        self.wave = WavePool(3)
        self.eps = 1e-6

    def forward(self, X, Y):
        LL,LH,HL,HH = self.wave(Y)
        Y_outs = [0.1*LL,0.2*LH,0.2*HL,0.5*HH]
        Yc = torch.cat(Y_outs,1)
        LL,LH,HL,HH  = self.wave(X)
        X_outs = [0.1*LL,0.2*LH,0.2*HL,0.5*HH]
        Xc = torch.cat(X_outs,1)
        diff = torch.add(Xc, -Yc)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss
# ----------------------------------------------------------------
class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))

# Mixed Link Block architecture
class MLB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(MLB, self).__init__()
        # delete

    def forward(self, x):
        return x

class SORG(nn.Module):
    def __init__(self):
        super(SORG, self).__init__()
        self.socarb1 = SOCARB(64)
        self.socarb2 = SOCARB(64)

    def forward(self, x):
        out = self.socarb1(x)
        out = self.socarb2(out)
        out = out+x
        return x


class SOCARB(nn.Module):
    # channel attention residual block
    def __init__(self, nChannels):
        super(SOCARB, self).__init__()
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.ca1 = SOCA(nChannels, 16)
        self.ca2 = SOCA(nChannels, 16)

    def forward(self, x):
        out = self.conv2(self.relu1(self.conv1(x)))
        b1 = self.ca1(out) +x

        out = self.relu2(self.conv3(b1))
        b2 = self.ca2(self.conv4(out)) +b1
        return b2

## second-order Channel attention (SOCA)
class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        # global average pooling: feature --> point
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        ##
        ## MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov*x

class CARB(nn.Module):
    # channel attention residual block
    def __init__(self, nChannels,reduction=16):
        super(CARB, self).__init__()
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.ca1 = FRM(nChannels, reduction)
        self.ca2 = FRM(nChannels, reduction)

    def forward(self, x):
        out = self.conv2(self.relu1(self.conv1(x)))
        b1 = self.ca1(out) +x

        out = self.relu2(self.conv3(b1))
        b2 = self.ca2(self.conv4(out)) +b1
        return b2

## Channel Attention (CA) Layer
class FRM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FRM, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAT(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAT, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.trans = nn.Conv2d(channel, channel//2, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        out = x * y
        out = self.trans(out)
        return out

class CATD(nn.Module):
    def __init__(self, channel, out_channels, reduction=16):
        super(CATD, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.trans = nn.Conv2d(channel, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        out = x * y
        out = self.trans(out)
        return out

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

class L1_Sobel_Loss(nn.Module):
    def __init__(self):
        super(L1_Sobel_Loss, self).__init__()
        # self.sobel = Sobel_dw(3)
        self.sobel = Sobel()
        self.eps = 1e-6

    def forward(self, demoire, image_target):
        edge_outputs = self.sobel(demoire)
        edge_Y = self.sobel(image_target)
        diff = torch.add(edge_outputs, -edge_Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

class L1_ASL(nn.Module):
    def __init__(self):
        super(L1_ASL, self).__init__()
        self.sobel = AdvancedSobel()
        self.eps = 1e-6

    def forward(self, demoire, image_target):
        edge_outputs = self.sobel(demoire)
        edge_Y = self.sobel(image_target)
        diff = torch.add(edge_outputs, -edge_Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

class L1_Sobel_Loss1(nn.Module):
    def __init__(self):
        super(L1_Sobel_Loss1, self).__init__()
        self.sobel = Sobel()
        self.eps = 1e-6

    def forward(self, edge_outputs, image_target):
        edge_Y = self.sobel(image_target)
        diff = torch.add(edge_outputs, -edge_Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class L1_Wavelet_Loss(nn.Module):
    def __init__(self):
        super(L1_Wavelet_Loss, self).__init__()
        self.wave = WavePool(3)
        self.eps = 1e-6

    def forward(self, X, Y):
        Y_outs = self.wave(Y)
        Yc = torch.cat(Y_outs,1)
        X = self.wave(X)
        Xc = torch.cat(X,1)
        diff = torch.add(Xc, -Yc)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

class L1_Wavelet_Loss1(nn.Module):
    def __init__(self):
        super(L1_Wavelet_Loss1, self).__init__()
        # self.sobel = Sobel_dw(3)
        self.wave = WavePool(3)
        self.eps = 1e-6

    def forward(self, X, Y):
        Y_outs = self.wave(Y)
        Yc = torch.cat(Y_outs,1)
        diff = torch.add(X, -Yc)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

## non_local module
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size,C,H,W = x.shape
        g_x = self.g(x)
        g_x = g_x.view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _dot_product(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _concatenation(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)

## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8,sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # nonlocal module
        self.non_local = (NONLocalBlock2D(in_channels=in_feat,inter_channels=inter_feat, sub_sample=sub_sample,bn_layer=bn_layer))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ## divide feature map into 4 part
        batch_size,C,H,W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)
        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]
        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd
        return  nonlocal_feat

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        self.features = nn.Sequential(
        
            # input is (1) x 128 x 128
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (64) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 16 x 16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        #self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.features(input)
        # print(out.shape)
        # state size. (512) x 8 x 8
        out = out.view(out.size(0), -1)
        
        # state size. (512 x 8 x 8)
        # print(out.shape)
        out = self.fc1(out)
        
        # state size. (1024)
        out = self.LeakyReLU(out)
        
        out = self.fc2(out)
        # state size. (1)

        out = out.mean(0)

        #out = self.sigmoid(out)
        return out.view(1) 
