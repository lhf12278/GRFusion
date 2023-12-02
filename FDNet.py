from __future__ import print_function, division
import torch.utils.data
from torchvision import transforms
from BaseModle import *

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, down=False):
        super(conv_block, self).__init__()
        if down:
            stride = 2
        else:
            stride = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.In = nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=True)
        self.selu = nn.SELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = F.interpolate(x, y.size()[2:], mode='bilinear', align_corners=True)
        z = self.conv(x)
        z = self.bn(z)
        z = self.relu(z)
        return z

#三分支
class MultiKerSize(nn.Module):
    def __init__(self, in_ch):
        super(MultiKerSize, self).__init__()
        self.sa = SpatialAttention()
        self.ca = Ca(in_ch)
        self.cross = CrossAttention_MP(dim=in_ch)

        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels=in_ch*2, out_channels=in_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.Tanh(),
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.Tanh(),
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(in_ch),
            nn.Tanh(),
        )

    def forward(self, x):
        fe_3x3_a = self.branch3x3(x)
        fe_3x3_a_sa = fe_3x3_a * self.sa(fe_3x3_a)
        fe_3x3_a_ca = fe_3x3_a * self.ca(fe_3x3_a)
        fe_3x3_a_mer = self.convlayer(torch.cat((fe_3x3_a_ca, fe_3x3_a_sa), dim=1))

        fe_5x5_a = self.branch5x5(x)
        fe_5x5_a_sa = fe_5x5_a * self.sa(fe_5x5_a)
        fe_5x5_a_ca = fe_5x5_a * self.ca(fe_5x5_a)
        fe_5x5_a_mer =self.convlayer(torch.cat((fe_5x5_a_sa, fe_5x5_a_ca), dim=1))

        fe_3x3_a = self.cross(fe_3x3_a_mer, fe_5x5_a_mer) + fe_3x3_a
        fe_5x5_a = self.cross(fe_5x5_a_mer, fe_3x3_a_mer) + fe_5x5_a

        fe_merge_a = torch.where(abs(fe_3x3_a) - abs(fe_5x5_a) >= 0, fe_3x3_a, fe_5x5_a)
        return fe_merge_a


class Alp(nn.Module):
    def __init__(self, in_ch):
        super(Alp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class Beta(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Beta, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class U_Net1(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=2):
        super(U_Net1, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.alpha = Alp(16)
        self.beta = Beta(16, 8)
        self.msmf1 = MultiKerSize(filters[0])
        self.msmf2 = MultiKerSize(filters[1])
        self.msmf3 = MultiKerSize(filters[2])
        self.msmf4 = MultiKerSize(filters[3])
        self.msmf5 = MultiKerSize(filters[4])
        self.togray = transforms.Grayscale(num_output_channels=1)
        self.in1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.FE_blurr = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, img1,  img2, img3=None, img4=None, img5=None, img6=None):
        if img3==None:
            ia = img1
            ib = img2
        elif img4 == None:
            ia = img1
            ib = torch.max(img2, img3)
        elif img5 == None:
            ia = img1
            ib = img2 + img3 + img4
        else:
            ia = img1
            ib = img2 + img3 + img4 + img5 + img6

        ia_reblurr = blur_2th(ia)
        ia_reblurr = normPRED(ia_reblurr)
        ia_rever = 1 - ia_reblurr
        ib_reblurr = blur_2th(ib)
        ib_reblurr = normPRED(ib_reblurr)
        ib_rever = 1 - ib_reblurr

        ia_re = self.FE_blurr(torch.cat((ia_reblurr, ib_rever), dim=1))
        ib_re = self.FE_blurr(torch.cat((ib_reblurr, ia_rever), dim=1))

        ia_layer1 = self.Conv1(ia)
        ib_layer1 = self.Conv1(ib)

        ia_modu1 = ia_layer1[:, :8, :, :]
        ia_modu2 = ia_layer1[:, 8:, :, :]
        ia_modul = self.alpha(ia_re) * ia_modu1 + self.beta(ia_re)
        ia_fe = torch.cat((ia_modul, ia_modu2), dim=1)

        ib_modu1 = ib_layer1[:, :8, :, :]
        ib_modu2 = ib_layer1[:, 8:, :, :]
        ib_modul = self.alpha(ib_re) * ib_modu1 + self.beta(ib_re)
        ib_fe = torch.cat((ib_modul, ib_modu2), dim=1)

        #layer1
        e1 = self.in1(torch.cat((ia_fe, ib_fe), dim=1))
        e1 = self.msmf1(e1)
        e2 = self.Maxpool1(e1)

        #layer2
        e2 = self.Conv2(e2)
        e2 = self.msmf2(e2)
        e3 = self.Maxpool2(e2)

        #layer3
        e3 = self.Conv3(e3)
        e3 = self.msmf3(e3)
        e4 = self.Maxpool3(e3)

        #layer4
        e4 = self.Conv4(e4)
        e4 = self.msmf4(e4)
        e5 = self.Maxpool4(e4)

        #layer5
        e5 = self.Conv5(e5)
        e5 = self.msmf5(e5)

        #d-layer5
        d5 = self.Up5(e5, e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5, e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4, e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3, e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        d1 = self.active(out)

        return d1
