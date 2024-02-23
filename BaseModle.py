import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from skimage import morphology
import torchvision.models as models

def blur_2th(img):
    B = img[:, 0, :, :].unsqueeze(1)
    G = img[:, 1, :, :].unsqueeze(1)
    R = img[:, 2, :, :].unsqueeze(1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B

    filtr = torch.tensor([[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]], device=img.device)
    assert img.ndim == 4 and (img.shape[1] == 1 or img.shape[1] == 3)
    filtr = filtr.expand(Y.shape[1], Y.shape[1], 3, 3)
    blur = F.conv2d(Y, filtr, bias=None, stride=1, padding=1)
    blur = F.conv2d(blur, filtr, bias=None, stride=1, padding=1)
    diff = torch.abs(Y - blur)
    diff = torch.cat((diff, diff, diff), dim=1)
    return diff

def fill(x):
    ar_a = x.cpu()
    ar_a = np.array(ar_a, dtype=bool)
    dst = morphology.remove_small_objects(ar_a, min_size=5000, connectivity=1)  # 去除面积小于300的连通域
    ar_a_neg = dst.astype(int)
    ar_a_neg = 1 - ar_a_neg
    ar_a_neg = np.array(ar_a_neg, dtype=bool)
    dst1 = morphology.remove_small_objects(ar_a_neg, min_size=5000, connectivity=1)
    dst1 = dst1.astype(int)
    fill_a = torch.Tensor(dst1).cuda()
    fill_a = 1 - fill_a
    return fill_a

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        print('loading resnet101...')
        # self.loss_network = models.resnet101(pretrained=True, num_classes=2, in_channel=2)
        self.loss_network = models.resnet101(pretrained=True)
        # Turning off gradient calculations for efficiency
        for param in self.loss_network.parameters():
            param.requires_grad = False
        # if is_cuda:
        #     self.loss_network.cuda()
        self.loss_network.cuda()
        print("done ...")

    def mse_loss(self, input, target):
        return torch.sum((input - target) ** 2) / input.data.nelement()
    def forward(self, output, label):
        self.perceptualLoss = self.mse_loss(self.loss_network(output),self.loss_network(label))
        return self.perceptualLoss

def find_unconsist(x, y, z=None, w=None, u=None, v=None):
    x_size = x.size()
    zeros = torch.zeros(x_size).cuda()
    ones = torch.ones(x_size).cuda()
    if z==None:
        m_sum = x + y
        m_consist = torch.where(m_sum == 1, ones, zeros)
        m_unconsist = 1 - m_consist
    elif w==None:
        m_sum = x + y + z
        m_consist = torch.where(m_sum == 1, ones, zeros)
        m_unconsist = 1 - m_consist
    elif u==None:
        m_sum = x + y + z + w
        m_consist = torch.where(m_sum == 1, ones, zeros)
        m_unconsist = 1 - m_consist
    else:
        m_sum = x + y + z + w + u + v
        m_consist = torch.where(m_sum == 1, ones, zeros)
        m_unconsist = 1 - m_consist

    return m_unconsist

def extract_patches(x, kernel_size=3, stride=1):
    x = x.float()
    if kernel_size != 1:
        # x = nn.ZeroPad2d(2)(x)
        x = nn.ReplicationPad2d(3)(x)
    x = x.permute(0, 2, 3, 1)
    x = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    return x.contiguous()

def combine(img1, img2, mask1, mask2, unconsist, out, img3=None,mask3=None,
            img4=None, mask4=None, img5=None, mask5=None, img6=None, mask6=None):
    mask1_size = mask1.size()
    zeros = torch.zeros(mask1_size, dtype=int).cuda()
    ones = torch.ones(mask1_size, dtype=int).cuda()

    if img3==None:
        mask1 = mask1 + 5*unconsist
        mask2 = mask2 + 5*unconsist
        mask1 = torch.where(mask1 == 1, ones, zeros)
        mask2 = torch.where(mask2 == 1, ones, zeros)
        mask3 = unconsist
        out_fused = mask1 * img1 + mask2 * img2 + mask3 * out
    elif img4==None:
        mask1 = mask1 + 5 * unconsist
        mask2 = mask2 + 5 * unconsist
        mask3 = mask3 + 5 * unconsist
        mask1 = torch.where(mask1 == 1, ones, zeros)
        mask2 = torch.where(mask2 == 1, ones, zeros)
        mask3 = torch.where(mask3 == 1, ones, zeros)
        mask4 = unconsist
        out_fused = mask1 * img1 + mask2 * img2 + mask3 * img3 + mask4 * out
    elif img5==None:
        mask1 = mask1 + 5 * unconsist
        mask2 = mask2 + 5 * unconsist
        mask3 = mask3 + 5 * unconsist
        mask4 = mask4 + 5 * unconsist
        mask1 = torch.where(mask1 == 1, ones, zeros)
        mask2 = torch.where(mask2 == 1, ones, zeros)
        mask3 = torch.where(mask3 == 1, ones, zeros)
        mask4 = torch.where(mask4 == 1, ones, zeros)
        mask5 = unconsist
        out_fused = mask1 * img1 + mask2 * img2 + mask3 * img3 + mask4 * img4 + mask5 * out
    else:
        mask1 = mask1 + 5 * unconsist
        mask2 = mask2 + 5 * unconsist
        mask3 = mask3 + 5 * unconsist
        mask4 = mask4 + 5 * unconsist
        mask5 = mask5 + 5 * unconsist
        mask6 = mask6 + 5 * unconsist
        mask1 = torch.where(mask1 == 1, ones, zeros)
        mask2 = torch.where(mask2 == 1, ones, zeros)
        mask3 = torch.where(mask3 == 1, ones, zeros)
        mask4 = torch.where(mask4 == 1, ones, zeros)
        mask5 = torch.where(mask5 == 1, ones, zeros)
        mask6 = torch.where(mask6 == 1, ones, zeros)
        mask7 = unconsist
        out_fused = mask1 * img1 + mask2 * img2 + mask3 * img3 + mask4 * img4 + mask5 * img5 + mask6 *img6 + mask7 * out

    return out_fused

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        # self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + res

        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class CrossAttention_MP(nn.Module):
    def __init__(
            self,
            dim,
            qkv_bias=False,
            qk_scale=1 / math.sqrt(64),
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.scale = qk_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        B1, C1, H1, W1 = x.shape
        x1 = x.permute(0, 2, 3, 1)
        qkv1 = self.qkv(x1)
        qkv1 = qkv1.reshape(B1, 3, C1, H1, W1).permute(1, 0, 2, 3, 4)
        q1, k1, v1 = (
            qkv1[0],
            qkv1[1],
            qkv1[2],
        )

        B2, C2, H2, W2 = y.shape
        y1 = y.permute(0, 2, 3, 1)
        qkv2 = self.qkv(y1)
        qkv2 = qkv2.reshape(B2, 3, C2, H2, W2).permute(1, 0, 2, 3, 4)
        q2, k2, v2 = (
            qkv2[0],
            qkv2[1],
            qkv2[2],
        )

        attn = torch.matmul(q2, k1.transpose(2, 3)) * self.scale

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        F = torch.matmul(attn, v1)
        return F

class Ca(nn.Module): #channel attention
    def __init__(self, in_planes, ratio=16):
        super(Ca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.softmax(out)
        return attn

class Selfpatch(object):
    def buildAutoencoder(self, target_img, target_img_2, target_img_3, patch_size=1, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, 'target image must be of dimension 3.'
        C = target_img.size(0)

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor

        patches_features = self._extract_patches(target_img, patch_size, stride)
        patches_features_f = self._extract_patches(target_img_3, patch_size, stride)

        patches_on = self._extract_patches(target_img_2, 1, stride)

        return patches_features_f, patches_features, patches_on

    def build(self, target_img,  patch_size=5, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, 'target image must be of dimension 3.'
        C = target_img.size(0)

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor

        patches_features = self._extract_patches(target_img, patch_size, stride)

        return patches_features

    def _build(self, patch_size, stride, C, target_patches, npatches, normalize, interpolate, type):
        # for each patch, divide by its L2 norm.
        if type == 1:
            enc_patches = target_patches.clone()
            for i in range(npatches):
                enc_patches[i] = enc_patches[i]*(1/(enc_patches[i].norm(2)+1e-8))

            conv_enc = nn.Conv2d(npatches, npatches, kernel_size=1, stride=stride, bias=False, groups=npatches)
            conv_enc.weight.data = enc_patches
            return conv_enc

        # normalize is not needed, it doesn't change the result!
            if normalize:
                raise NotImplementedError

            if interpolate:
                raise NotImplementedError
        else:

            conv_dec = nn.ConvTranspose2d(npatches, C, kernel_size=patch_size, stride=stride, bias=False)
            conv_dec.weight.data = target_patches
            return conv_dec

    def _extract_patches(self, img, patch_size, stride):
        n_dim = 3
        assert img.dim() == n_dim, 'image must be of dimension 3.'
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = input_windows.size(0), input_windows.size(1), input_windows.size(2), input_windows.size(3), input_windows.size(4)
        input_windows = input_windows.permute(1,2,0,3,4).contiguous().view(i_2*i_3, i_1, i_4, i_5)
        patches_all = input_windows
        return patches_all


def corr_fun(Kernel_tmp, Feature):
    size = Kernel_tmp.size()
    CORR = []
    Kernel = []
    for i in range(len(Feature)):
        ker = Kernel_tmp[i:i + 1]
        fea = Feature[i:i + 1]
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
        ker = ker.unsqueeze(2).unsqueeze(3)

        co = F.conv2d(fea, ker.contiguous())
        CORR.append(co)
        ker = ker.unsqueeze(0)
        Kernel.append(ker)
    corr = torch.cat(CORR, 0)
    Kernel = torch.cat(Kernel, 0)
    return corr, Kernel

class CorrelationLayer(nn.Module):
    def __init__(self, corr_size, feat_channel):
        super(CorrelationLayer, self).__init__()

        self.pool_layer = nn.AdaptiveAvgPool2d((corr_size, corr_size))

        self.corr_reduce = nn.Sequential(
            nn.Conv2d(corr_size * corr_size, feat_channel, kernel_size=1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(),
            nn.Conv2d(feat_channel, feat_channel, 3, padding=1),
        )
        self.Dnorm = nn.InstanceNorm2d(feat_channel)

        self.feat_adapt = nn.Sequential(
            # nn.Conv2d(feat_channel * 2, feat_channel, 1),
            nn.Conv2d(320, feat_channel, 1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # calculate correlation map
        RGB_feat_downsize = F.normalize(self.pool_layer(x))
        RGB_feat_norm = F.normalize(x)
        RGB_corr, _ = corr_fun(RGB_feat_downsize, RGB_feat_norm)

        Red_corr = self.corr_reduce(RGB_corr)

        # beta cond
        new_feat = torch.cat((x, Red_corr), dim=1)
        new_feat = self.feat_adapt(new_feat)

        # Depth_feat = self.Dnorm(x[1])
        return new_feat