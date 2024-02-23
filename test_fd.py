import os
import torch
import time
import torch.nn.functional as F
from FDNet import U_Net1
from Dataloader import *
from BaseModle import *
from skimage import morphology

# print(torch.cuda.current_device())
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ids = torch.cuda.device_count()
device = torch.device('cuda:0')       # CUDA:0

fd = U_Net1(3, 2)
fd_path = "/checkpoints1/kernel_37.pth"
use_gpu = torch.cuda.is_available()
# use_gpu = False


if use_gpu:
    print('GPU Mode Acitavted')
    fd = fd.cuda()
    fd.cuda()
    # device_ids = range(torch.cuda.device_count())
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    fd.load_state_dict(torch.load(fd_path))
    # print(model)
else:
    print('CPU Mode Acitavted')
    state_dict = torch.load(fd_path, map_location='cpu')
    # load params
    fd.load_state_dict(state_dict)

# normalize the predicted probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def extract_patches(x, kernel_size=3, stride=1):
    x = x.float()
    if kernel_size != 1:
        # x = nn.ZeroPad2d(2)(x)
        x = nn.ReplicationPad2d(3)(x)
    x = x.permute(0, 2, 3, 1)
    x = x.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
    return x.contiguous()

def fusion_color(file_path, type, save_path, couples, img_num):
    fd.eval()
    if img_num == 2:
        with torch.no_grad():
            for num in range(1, couples + 1):
                tic = time.time()
                path1 = file_path + '/lytro_{}{}_A.'.format(num//10, num % 10) + type           # for the "Lytro" dataset
                path2 = file_path + '/lytro_{}{}_B.'.format(num//10, num % 10) + type           # for the "Lytro" dataset
                # path1 = file_path + '/MFI-WHU_{}{}_A.'.format(num // 10, num % 10) + type  # for the "MFFW"dataset
                # path2 = file_path + '/MFI-WHU_{}{}_B.'.format(num // 10, num % 10) + type  # for the "MFFW" dataset
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
                img1_read = np.array(img1)
                img2_read = np.array(img2)  # R G B
                h = img1_read.shape[0]
                w = img1_read.shape[1]
                img1_org = img1
                img2_org = img2
                tran = transforms.Compose([transforms.ToTensor()])
                img1 = tran(img1_org)
                img2 = tran(img2_org)
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)

                if use_gpu:
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                else:
                    img1 = img1
                    img2 = img2

                outa = fd(img1, img2)
                out1 = outa[:, 0, :, :].unsqueeze(1)
                out2 = outa[:, 1, :, :].unsqueeze(1)
                confidence_map1 = torch.max(out1, out2)

                outb = fd(img2, img1)
                out11 = outb[:, 0, :, :].unsqueeze(1)
                out22 = outb[:, 1, :, :].unsqueeze(1)
                confidence_map2 = torch.max(out11, out22)

                img1_mask_bi = to_binary(confidence_map1)
                img2_mask_bi = to_binary(confidence_map2)

                unconsis = find_unconsist(img1_mask_bi, img2_mask_bi)

                img1_mask_bi = img1_mask_bi.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img2_mask_bi = img2_mask_bi.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                unconsis_bi = unconsis.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()

                img1_mask_bi = Image.fromarray(np.uint8(img1_mask_bi))
                img2_mask_bi = Image.fromarray(np.uint8(img2_mask_bi))
                unconsis_bi = Image.fromarray(np.uint8(unconsis_bi))

                img1_mask_bi.save(save_path + '/MFI-WHU_{}{}_A.jpg'.format(num // 10, num % 10))
                img2_mask_bi.save(save_path + '/MFI-WHU_{}{}_B.jpg'.format(num // 10, num % 10))
                # img1_mask_bi.save(save_path + '/Lytro_{}{}_A.jpg'.format(num // 10, num % 10))
                # img2_mask_bi.save(save_path + '/Lytro_{}{}_B.jpg'.format(num // 10, num % 10))

                toc = time.time()
                print('end lytro_{}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))

    elif img_num == 3:
        with torch.no_grad():

            for num in range(1, couples + 1):
                tic = time.time()
                path1 = file_path + '/lytro-{}{}-A.'.format(num//10, num % 10) + type           # for the "Lytro" dataset
                path2 = file_path + '/lytro-{}{}-B.'.format(num//10, num % 10) + type           # for the "Lytro" dataset
                path3 = file_path + '/lytro-{}{}-C.'.format(num//10, num % 10) + type           # for the "Lytro" dataset
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
                img3 = Image.open(path3).convert('RGB')
                img1_read = np.array(img1)
                img2_read = np.array(img2)  # R G B
                img3_read = np.array(img3)  # R G B
                h = img1_read.shape[0]
                w = img1_read.shape[1]
                img1_org = img1
                img2_org = img2
                img3_org = img3
                tran = transforms.Compose([transforms.ToTensor()])
                img1 = tran(img1_org)
                img2 = tran(img2_org)
                img3 = tran(img3_org)
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
                img3 = img3.unsqueeze(0)

                if use_gpu:
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                    img3 = img3.cuda()
                else:
                    img1 = img1
                    img2 = img2
                    img3 = img3

                outa = fd(img1, img2, img3)
                out1 = outa[:, 0, :, :].unsqueeze(1)
                out2 = outa[:, 1, :, :].unsqueeze(1)
                confidence_map1 = torch.max(out1, out2)

                outb = fd(img2, img1, img3)
                out11 = outb[:, 0, :, :].unsqueeze(1)
                out22 = outb[:, 1, :, :].unsqueeze(1)
                confidence_map2 = torch.max(out11, out22)

                outc = fd(img3, img1, img2)
                out111 = outc[:, 0, :, :].unsqueeze(1)
                out222 = outc[:, 1, :, :].unsqueeze(1)
                confidence_map3 = torch.max(out111, out222)

                out1 = to_binary(confidence_map1)
                out2 = to_binary(confidence_map2)
                out3 = to_binary(confidence_map3)

                out1 = fill(out1)
                out2 = fill(out2)
                out3 = fill(out3)

                unconsis = find_unconsist(out1, out2, out3)

                img1_mask_bi = out1.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img2_mask_bi = out2.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img3_mask_bi = out3.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img_un = unconsis.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()

                img1_mask_bi = Image.fromarray(np.uint8(img1_mask_bi))
                img2_mask_bi = Image.fromarray(np.uint8(img2_mask_bi))
                img3_mask_bi = Image.fromarray(np.uint8(img3_mask_bi))
                img_un = Image.fromarray(np.uint8(img_un))
                img1_mask_bi.save(save_path + '/triple_sets_{}{}_A.jpg'.format(num // 10, num % 10))
                img2_mask_bi.save(save_path + '/triple_sets_{}{}_B.jpg'.format(num // 10, num % 10))
                img3_mask_bi.save(save_path + '/triple_sets_{}{}_C.jpg'.format(num // 10, num % 10))
                img_un.save(save_path + '/triple_sets_{}{}_un.jpg'.format(num // 10, num % 10))

                toc = time.time()
                print('end lytro_{}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))

    elif img_num == 4:
        with torch.no_grad():
            for num in range(1, couples + 1):
                tic = time.time()
                path1 = file_path + '/mffw-{}{}-1.'.format(num // 10, num % 10) + type
                path2 = file_path + '/mffw-{}{}-2.'.format(num // 10, num % 10) + type
                path3 = file_path + '/mffw-{}{}-3.'.format(num // 10, num % 10) + type
                path4 = file_path + '/mffw-{}{}-4.'.format(num // 10, num % 10) + type
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
                img3 = Image.open(path3).convert('RGB')
                img4 = Image.open(path4).convert('RGB')
                img1_read = np.array(img1)
                img2_read = np.array(img2)  # R G B
                img3_read = np.array(img3)  # R G B
                img4_read = np.array(img4)  # R G B
                h = img1_read.shape[0]
                w = img1_read.shape[1]
                img1_org = img1
                img2_org = img2
                img3_org = img3
                img4_org = img4
                tran = transforms.Compose([transforms.ToTensor()])
                img1 = tran(img1_org)
                img2 = tran(img2_org)
                img3 = tran(img3_org)
                img4 = tran(img4_org)
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
                img3 = img3.unsqueeze(0)
                img4 = img4.unsqueeze(0)

                if use_gpu:
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                    img3 = img3.cuda()
                    img4 = img4.cuda()
                else:
                    img1 = img1
                    img2 = img2
                    img3 = img3
                    img4 = img4

                outa = fd(img1, img2, img3, img4)
                out1 = outa[:, 0, :, :].unsqueeze(1)
                out2 = outa[:, 1, :, :].unsqueeze(1)
                confidence_map1 = torch.max(out1, out2)

                outb = fd(img2, img1, img3, img4)
                out11 = outb[:, 0, :, :].unsqueeze(1)
                out22 = outb[:, 1, :, :].unsqueeze(1)
                confidence_map2 = torch.max(out11, out22)

                outc = fd(img3, img1, img2, img4)
                out111 = outc[:, 0, :, :].unsqueeze(1)
                out222 = outc[:, 1, :, :].unsqueeze(1)
                confidence_map3 = torch.max(out111, out222)

                outd = fd(img4, img1, img2, img3)
                out111 = outd[:, 0, :, :].unsqueeze(1)
                out222 = outd[:, 1, :, :].unsqueeze(1)
                confidence_map4 = torch.max(out111, out222)

                out1 = to_binary(confidence_map1)
                out2 = to_binary(confidence_map2)
                out3 = to_binary(confidence_map3)
                out4 = to_binary(confidence_map4)

                img1_mask_bi = out1.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img2_mask_bi = out2.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img3_mask_bi = out3.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img4_mask_bi = out4.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                # img_un = unconsis.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()

                img1_mask_bi = Image.fromarray(np.uint8(img1_mask_bi))
                img2_mask_bi = Image.fromarray(np.uint8(img2_mask_bi))
                img3_mask_bi = Image.fromarray(np.uint8(img3_mask_bi))
                img4_mask_bi = Image.fromarray(np.uint8(img4_mask_bi))
                # img_un = Image.fromarray(np.uint8(img_un))
                img1_mask_bi.save(save_path + '/mffw_{}{}_A.jpg'.format(num // 10, num % 10))
                img2_mask_bi.save(save_path + '/mffw_{}{}_B.jpg'.format(num // 10, num % 10))
                img3_mask_bi.save(save_path + '/mffw_{}{}_C.jpg'.format(num // 10, num % 10))
                img4_mask_bi.save(save_path + '/mffw_{}{}_D.jpg'.format(num // 10, num % 10))
                # img_un.save(save_path + '/triple_sets_{}{}_un.jpg'.format(num // 10, num % 10))

                toc = time.time()
                print('end lytro_{}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))
    else:
        with torch.no_grad():
            for num in range(1, couples + 1):
                tic = time.time()
                path1 = file_path + '/mffw-{}{}-1.'.format(num // 10, num % 10) + type
                path2 = file_path + '/mffw-{}{}-2.'.format(num // 10, num % 10) + type
                path3 = file_path + '/mffw-{}{}-3.'.format(num // 10, num % 10) + type
                path4 = file_path + '/mffw-{}{}-4.'.format(num // 10, num % 10) + type
                path5 = file_path + '/mffw-{}{}-5.'.format(num // 10, num % 10) + type
                path6 = file_path + '/mffw-{}{}-6.'.format(num // 10, num % 10) + type
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
                img3 = Image.open(path3).convert('RGB')
                img4 = Image.open(path4).convert('RGB')
                img5 = Image.open(path5).convert('RGB')
                img6 = Image.open(path6).convert('RGB')

                img1_read = np.array(img1)
                img2_read = np.array(img2)
                img3_read = np.array(img3)
                img4_read = np.array(img4)
                img5_read = np.array(img5)
                img6_read = np.array(img6)

                h = img1_read.shape[0]
                w = img1_read.shape[1]
                img1_org = img1
                img2_org = img2
                img3_org = img3
                img4_org = img4
                img5_org = img5
                img6_org = img6
                tran = transforms.Compose([transforms.ToTensor()])
                img1 = tran(img1_org)
                img2 = tran(img2_org)
                img3 = tran(img3_org)
                img4 = tran(img4_org)
                img5 = tran(img5_org)
                img6 = tran(img6_org)
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
                img3 = img3.unsqueeze(0)
                img4 = img4.unsqueeze(0)
                img5 = img5.unsqueeze(0)
                img6 = img6.unsqueeze(0)

                if use_gpu:
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                    img3 = img3.cuda()
                    img4 = img4.cuda()
                    img5 = img5.cuda()
                    img6 = img6.cuda()
                else:
                    img1 = img1
                    img2 = img2
                    img3 = img3
                    img4 = img4
                    img5 = img5
                    img6 = img6

                outa = fd(img1, img2, img3, img4, img5, img6)
                out11 = outa[:, 0, :, :].unsqueeze(1)
                out12 = outa[:, 1, :, :].unsqueeze(1)
                confidence_map1 = torch.max(out11, out12)

                outb = fd(img2, img1, img3, img4, img5, img6)
                out21 = outb[:, 0, :, :].unsqueeze(1)
                out22 = outb[:, 1, :, :].unsqueeze(1)
                confidence_map2 = torch.max(out21, out22)

                outc = fd(img3, img1, img2, img4, img5, img6)
                out31 = outc[:, 0, :, :].unsqueeze(1)
                out32 = outc[:, 1, :, :].unsqueeze(1)
                confidence_map3 = torch.max(out31, out32)

                outd = fd(img4, img1, img2, img3, img5, img6)
                out41 = outd[:, 0, :, :].unsqueeze(1)
                out42 = outd[:, 1, :, :].unsqueeze(1)
                confidence_map4 = torch.max(out41, out42)

                oute = fd(img5, img1, img2, img3, img4, img6)
                out51 = oute[:, 0, :, :].unsqueeze(1)
                out52 = oute[:, 1, :, :].unsqueeze(1)
                confidence_map5 = torch.max(out51, out52)

                outf = fd(img6, img1, img2, img3, img4, img5)
                out61 = outf[:, 0, :, :].unsqueeze(1)
                out62 = outf[:, 1, :, :].unsqueeze(1)
                confidence_map6 = torch.max(out61, out62)

                out1 = to_binary(confidence_map1)
                out2 = to_binary(confidence_map2)
                out3 = to_binary(confidence_map3)
                out4 = to_binary(confidence_map4)
                out5 = to_binary(confidence_map5)
                out6 = to_binary(confidence_map6)

                img1_mask_bi = out1.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img2_mask_bi = out2.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img3_mask_bi = out3.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img4_mask_bi = out4.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img5_mask_bi = out5.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()
                img6_mask_bi = out6.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).squeeze(0).cpu()

                img1_mask_bi = Image.fromarray(np.uint8(img1_mask_bi))
                img2_mask_bi = Image.fromarray(np.uint8(img2_mask_bi))
                img3_mask_bi = Image.fromarray(np.uint8(img3_mask_bi))
                img4_mask_bi = Image.fromarray(np.uint8(img4_mask_bi))
                img5_mask_bi = Image.fromarray(np.uint8(img5_mask_bi))
                img6_mask_bi = Image.fromarray(np.uint8(img6_mask_bi))
                img1_mask_bi.save(save_path + '/triple_sets_{}{}_A.jpg'.format(num // 10, num % 10))
                img2_mask_bi.save(save_path + '/triple_sets_{}{}_B.jpg'.format(num // 10, num % 10))
                img3_mask_bi.save(save_path + '/triple_sets_{}{}_C.jpg'.format(num // 10, num % 10))
                img4_mask_bi.save(save_path + '/triple_sets_{}{}_D.jpg'.format(num // 10, num % 10))
                img5_mask_bi.save(save_path + '/triple_sets_{}{}_E.jpg'.format(num // 10, num % 10))
                img6_mask_bi.save(save_path + '/triple_sets_{}{}_F.jpg'.format(num // 10, num % 10))

                toc = time.time()
                print('end lytro_{}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))

if __name__ == '__main__':
    fusion_color('../Mfif/Lytro', 'jpg', './test_fusion', 20, 2)     # fuse the "Lytro" dataset