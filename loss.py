import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision.models as models

class TextureLoss(nn.Module):
    def __init__(self):
        super(TextureLoss, self).__init__()
        #vgg = vgg16(pretrained=True)
        model = models.vgg16(pretrained=True)
        #print(model)
        loss_network = nn.Sequential(*list(model.features)[:24]).eval()
        loss_network = loss_network.cuda()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, f_img, v_img):
        # Perception Loss
        # f_i = f_img.repeat([1, 3, 1, 1])
        # v_i = v_img[0].repeat([1, 3, 1, 1])
        f_feature = self.loss_network(f_img)
        v_feature = self.loss_network(v_img)

        # gram Loss
        #gram_loss = self.mse_loss(self.gram_matrix(f_feature), self.gram_matrix(v_feature))
        perceptual_loss = self.mse_loss(f_feature, v_feature)

        return perceptual_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :h - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w - 1]), 2).mean()
        return tv1 + tv2


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

#--------------------------perceptual loss-----------------------------------

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # pre = torch.load('/root/wd/code/Mfif/Mfif1/data/vgg19-dcbb9e9d.pth')
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.criterion1 = nn.MSELoss()
        # self.weights = [1.0/32, 1.0/32, 1.0/16, 1.0/16, 13/16]
        self.weights = [1.0 / 16, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion1(x_vgg[i], y_vgg[i].detach())

        return loss

#-------------------------gradient loss-------------------------------

def gradient_compute(real_image, generated): # b x c x h x w
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    critertion11 = nn.L1Loss()
    critertion11 = critertion11.to(device)
    critertion12 = nn.L1Loss()
    critertion12 = critertion12.to(device)

    true_x_shifted_right = real_image[:, :, 1:, :]# 32 x 3 x 255 x 256
    true_x_shifted_left = real_image[:, :, :-1, :]
    true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

    generated_x_shift_right = generated[:, :, 1:, :]# 32 x 3 x 255 x 256
    generated_x_shift_left = generated[:, :, :-1, :]
    generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

    # loss_x = critertion10(generated_x_griednt)
    # difference_x = true_x_gradient - generated_x_griednt
    # loss_x_gradient = (torch.sum(difference_x)**2)/2 # tf.nn.l2_loss(true_x_gradient - generated_x_gradient)
    loss_x = critertion11(generated_x_griednt, true_x_gradient)

    true_y_shifted_right = real_image[:, :, :, 1:]
    true_y_shifted_left = real_image[:, :, :, :-1]
    true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

    generated_y_shift_right = generated[:, :, :, 1:]
    generated_y_shift_left = generated[:, :, :, :-1]
    generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

    loss_y = critertion12(generated_y_griednt, true_y_gradient)

    loss_gra = loss_y + loss_x
    # difference_y = true_y_gradient - generated_y_griednt
    # loss_y_gradient = (torch.sum(difference_y)**2)/2 # tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    return loss_gra

#-------------------------lp -------------------------------
class LpLssimLoss(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        """
            Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)   # [window_size, 1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # [1,1,window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, image_in, image_out):

        # Check if need to create the gaussian window
        (_, channel, _, _) = image_in.size()
        if channel == self.channel and self.window.data.type() == image_in.data.type():
            pass
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(image_out.get_device())
            window = window.type_as(image_in)
            self.window = window
            self.channel = channel

        # Lp
        Lp = torch.sqrt(torch.sum(torch.pow((image_in - image_out), 2)))  # 二范数
        # Lp = torch.sum(torch.abs(image_in - image_out))  # 一范数
        # Lssim
        Lssim = 1 - self._ssim(image_in, image_out, self.window, self.window_size, self.channel, self.size_average)
        return Lp + Lssim * 1000, Lp, Lssim * 1000

