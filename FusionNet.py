from BaseModle import *

# weight generator
class WG(nn.Module):
    def __init__(self, in_channel):
        super(WG, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channel//2, in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channel//4, in_channel//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel//4, in_channel//4, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channel//4, in_channel//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel//4, in_channel//4, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channel//4, in_channel//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel//4, in_channel//4, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(in_channel//4),
            nn.ReLU()
        )
        self.layer1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, merge):
        ia_ib = merge
        branch1 = self.branch1(ia_ib)
        branch2 = self.branch2(ia_ib)
        branch3 = self.branch2(ia_ib)
        branch4 = self.branch2(ia_ib)
        branch5 = self.branch2(ia_ib)
        branch_cat = torch.cat((branch2, branch3, branch4, branch5), dim=1)
        x = branch1 + branch_cat
        x = self.layer1(x)
        return x

# multidirectioanl edge extract
class MDEE(nn.Module):
    def __init__(self, c_in=64):
        super(MDEE, self).__init__()
        self.layer = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.WG = WG(c_in)
        self.softmax = nn.Softmax(dim=1)
        self.ConvLayer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(8),
        )

        #0
        self.acr0 = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        self.acr45 = [[0, 0, -1], [0, 1, 0], [0, 0, 0]]
        self.acr90 = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
        self.acr135 = [[-1, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.acr180 = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
        self.acr225 = [[0, 0, 0], [0, 1, 0], [-1, 0, 0]]
        self.acr270 = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        self.acr315 = [[0, 0, 0], [0, 1, 0], [0, 0, -1]]

    def forward(self, fe, guide_fe):
        input = fe
        b, c, h, w = input.shape
        B = input[:, 0, :, :].unsqueeze(1)
        G = input[:, 1, :, :].unsqueeze(1)
        R = input[:, 2, :, :].unsqueeze(1)
        input = 0.299 * R + 0.587 * G + 0.114 * B

        #概率生成
        glo = self.WG(guide_fe)
        prob = self.ConvLayer(glo)

        x_chunk = input.chunk(b, dim=0)
        prob_chunk = prob.chunk(b, dim=0)

        acr0 = torch.tensor(self.acr0)
        acr45 = torch.tensor(self.acr45)
        acr90 = torch.tensor(self.acr90)
        acr135 = torch.tensor(self.acr135)
        acr180 = torch.tensor(self.acr180)
        acr225 = torch.tensor(self.acr225)
        acr270 = torch.tensor(self.acr270)
        acr315 = torch.tensor(self.acr315)

        acr0 = acr0.cuda().float()
        acr45 = acr45.cuda().float()
        acr90 = acr90.cuda().float()
        acr135 = acr135.cuda().float()
        acr180 = acr180.cuda().float()
        acr225 = acr225.cuda().float()
        acr270 = acr270.cuda().float()
        acr315 = acr315.cuda().float()

        k_list = []
        k_list.append(acr0)
        k_list.append(acr45)
        k_list.append(acr90)
        k_list.append(acr135)
        k_list.append(acr180)
        k_list.append(acr225)
        k_list.append(acr270)
        k_list.append(acr315)

        feature = []
        for idx in range(b):
            fea = x_chunk[idx]
            prob = prob_chunk[idx]
            b1, c1, h1, w1 = fea.shape
            y_l = []
            for num in range(8):
                p = k_list[num].repeat(c1, 1, 1, 1)
                weight = nn.Parameter(p, requires_grad=False)
                y = F.conv2d(fea.contiguous(), weight=weight, groups=c1, padding=1)
                new_y = abs(y)
                new_y = torch.cat((new_y, new_y, new_y), dim=1)
                y_l.append(new_y)

            prob_k = self.softmax(prob)
            prob = prob_k.chunk(8, dim=1)

            for m in range(8):
                move_after = y_l[m] * prob[m]
                move_after += move_after
            feature.append(move_after)

        fe = [aa.tolist() for aa in feature]
        fe = torch.tensor(fe)
        move = fe.view(b, c, h, w)
        move = move.cuda().float()
        return move

# 图像重建（恢复）模块
class ImgRebuilt(nn.Module):
    def __init__(self, ngf=16):
        super(ImgRebuilt, self).__init__()
        self.conv1 = nn.Conv2d(64, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ngf * 4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ngf * 4, ngf, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ngf)
        self.conv3 = nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(65, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.resB1 = ResBlock(65)
        self.resB2 = ResBlock(96)
        self.resB3 = ResBlock(64)
        self.resB4 = ResBlock(64)
        self.resB5 = ResBlock(64)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.resB1(x)
        x = self.layer1(x)
        x = self.resB2(x)
        x = self.layer2(x)
        x = self.resB3(x)
        x = self.resB4(x)
        x = self.resB5(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

# Edge Feature Embeding
class EFE(nn.Module):
    def __init__(self):
        super(EFE, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.ps = 5
        self.en_01 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.en_002 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.en_02 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1)

    def forward(self, ia, ib):
        Nonparm = Selfpatch()

        #the upper path
        x_id = self.en_01(ib)
        x_vg = self.en_002(ia)
        x_rs = self.en_02(ib)

        b, c, h, w = x_rs.size()
        # print(x.size())

        x_rs = torch.sigmoid(x_rs)
        x_vg = torch.sigmoid(x_vg)

        k = (self.ps - 1)//2

        csa2_f = torch.nn.functional.pad(x_rs, (k, k, k, k))
        csa2_ff = torch.nn.functional.pad(x_id, (k, k, k, k))
        # print(csa2_f.size(), csa2_ff.size())

        csa2_f = csa2_f.view(-1, h+2*k, w+2*k) #b c 合并
        x_vg = x_vg.view(-1, h, w)
        csa2_ff = csa2_ff.view(-1, h+2*k, w+2*k)
        # print(csa2_ff.size(), csa2_f.size(), x_vg.size())

        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(csa2_f, x_vg, csa2_ff, self.ps, 1)
        # print(csa2_fff.size(), csa2_f.size(), csa2_conv.size())

        csa2_fff = csa2_fff.view(-1, 64, self.ps, self.ps)
        csa2_f = csa2_f.view(-1, 64, self.ps, self.ps)
        csa2_conv = csa2_conv.view(-1, 64, 1, 1)

        csa2_conv = F.normalize(csa2_conv, p=2, dim=1)
        csa2_f = F.normalize(csa2_f, p=2, dim=1)
        # print(csa2_f.size(), csa2_conv.size())

        csa2_conv = csa2_conv.expand_as(csa2_f)
        csa_a = csa2_conv * csa2_f
        csa_a = torch.mean(csa_a, 1)
        a_c, a_h, a_w = csa_a.size()
        # print(csa_a.size())
        csa_a = csa_a.contiguous().view(a_c, -1)
        csa_a = F.softmax(csa_a, dim=1)
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
        out1 = csa_a * csa2_fff
        out1 = torch.sum(out1, -1)
        out1 = torch.sum(out1, -1)
        out_csa = out1.contiguous().view(b, c, h, w)
        out_32 = self.down(out_csa)
        out_32 = self.layer1(out_32)
        return out_32

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.MDEE = MDEE()
        self.EFE = EFE()
        self.sa = SpatialAttention()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels=67, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.FE1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, ia):
        img1_fe = self.FE(ia)
        img1_fe_1 = self.FE1(ia)
        fe_edg_1 = self.MDEE(ia, img1_fe_1)
        fe_1 = self.EFE(fe_edg_1, img1_fe) + img1_fe
        return fe_1

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.ImgRebuilt = ImgRebuilt()

    def forward(self, fe_max, mask_unconsist):
        out = self.ImgRebuilt(fe_max, 2*mask_unconsist)
        return out
