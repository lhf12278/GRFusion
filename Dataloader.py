from __future__ import print_function
import torch.utils.data as Data
import torchvision.transforms
import torchvision.transforms as transforms
import numpy as np
from glob import glob
import os
import cv2
import torch
import copy
from PIL import Image
import random
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.8, aug)
np.random.seed(2)

class ToTensor(object):
    def __call__(self, data_dict):
        for key in data_dict:
            d = data_dict[key]
            d = np.ascontiguousarray(np.transpose(d, (2, 0, 1)))
            data_dict[key] = torch.from_numpy(d).float()/255

        return data_dict

class Fusion_data(Data.Dataset):
    def __init__(self, io, args, root, transform=None, gray=True, partition='train'):
        self.files = glob(os.path.join(root, '*.*'))
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transforms.Compose([ToTensor()])
        self.args = args
        self.ps = 256

        if args.miniset == True:
            self.files = random.sample(self.files, int(args.minirate * len(self.files)))
        self.num_examples = len(self.files)

        if partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)
        io.cprint("number of " + partition + " examples in dataset" + ": " + str(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        Img = cv2.imread(self.files[index])
        ia, ib, ia_blurr, ib_blurr, ia_re, ib_re, fuse_label, ia_mask_label, ib_mask_label = self.get_patch(Img)
        sample = {'ia': ia, 'ib': ib, 'ia_blurr': ia_blurr, 'ib_blurr': ib_blurr, 'ia_re': ia_re, 'ib_re': ib_re,
                  'fuse_label': fuse_label, 'ia_mask_label': ia_mask_label, 'ib_mask_label': ib_mask_label}
        sample = self.transform(sample)

        return sample

    def get_patch(self, Img):
        H, W = Img.shape[0], Img.shape[1]//2
        img, mask = Img[:, :W, :], Img[:, W:, :]//255

        #1.
        x, y = np.random.randint(10, H-10-self.ps+1), np.random.randint(10, W-10-self.ps+1)
        fuse_label = img[x:x+self.ps, y:y+self.ps, :]

        ia_mask_label = mask[x:x + self.ps, y:y + self.ps, 0][:, :, None]
        ib_mask_label = 1 - ia_mask_label

        #5.
        level = np.random.randint(0, 5)
        if level == 0:
            fuse_label_Blur = cv2.GaussianBlur(fuse_label, (3, 3), sigmaX=1, sigmaY=1)
        if level == 1:
            fuse_label_Blur = cv2.GaussianBlur(fuse_label, (5, 5), sigmaX=1.5, sigmaY=1.5)
        if level == 2:
            fuse_label_Blur = cv2.GaussianBlur(fuse_label, (7, 7), sigmaX=2, sigmaY=2)
        if level == 3:
            fuse_label_Blur = cv2.GaussianBlur(fuse_label, (11, 11), sigmaX=2.5, sigmaY=2.5)
        if level == 4:
            fuse_label_Blur = cv2.GaussianBlur(fuse_label, (15, 15), sigmaX=3, sigmaY=3)


        ia = fuse_label*ia_mask_label + fuse_label_Blur*(1-ia_mask_label)
        ib = fuse_label*ib_mask_label + fuse_label_Blur*(1-ib_mask_label)

        ia_blurr = cv2.GaussianBlur(ia, (3, 3), sigmaX=1, sigmaY=1)
        ib_blurr = cv2.GaussianBlur(ib, (3, 3), sigmaX=1, sigmaY=1)

        ia_re = ia - ia_blurr
        ib_re = ib - ib_blurr

        return ia, ib, ia_blurr, ib_blurr, ia_re, ib_re, fuse_label, ia_mask_label*255, ib_mask_label*255


def to_binary(mask):
    ones = torch.ones_like(mask)
    zero = torch.zeros_like(mask)

    a = torch.where(mask >= 0.5, ones, mask)
    binary = torch.where(a < 0.5, zero, a)

    return binary

def incrase_C(single_C_list):
    for i in range(len(single_C_list)):
        single_C_list[i] = torch.cat([single_C_list[i]] * 3, 1)

    return single_C_list



