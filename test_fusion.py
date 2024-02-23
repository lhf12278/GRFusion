import time
from FDNet import U_Net1
from FusionNet1 import Encoder, Decoder
from Dataloader import *
from BaseModle import *

# print(torch.cuda.current_device())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# ids = torch.cuda.device_count()
device = torch.device('cuda:0')       # CUDA:0

fd = U_Net1(3, 2)
fd_path = "./checkpoints/kernel_37.pth"   # MSNet_cascade_1e-3_8_80_BCE
use_gpu = torch.cuda.is_available()
# use_gpu = False

encoder = Encoder()
encoder_path = "./checkpoints/Encoder0.05.pth"
use_gpu = torch.cuda.is_available()

decoder = Decoder()
decoder_path = "./checkpoints/Decoder0.05.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
    fd = fd.cuda()
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    fd.cuda()
    encoder.cuda()
    decoder.cuda()
    # device_ids = range(torch.cuda.device_count())
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    fd.load_state_dict(torch.load(fd_path))
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    # print(model)
else:
    print('CPU Mode Acitavted')
    state_dict = torch.load(fd_path, map_location='cpu')
    # load params
    fd.load_state_dict(state_dict)

def fusion_color(file_path, type, save_path, couples, img_nums):
    fd.eval()
    encoder.eval()
    decoder.eval()
    if img_nums == 2:
        with torch.no_grad():
            for num in range(1, couples + 1):
                tic = time.time()
                path1 = file_path + '/lytro_{}{}_A.'.format(num // 10, num % 10) + type  # for the "Lytro" dataset
                path2 = file_path + '/lytro_{}{}_B.'.format(num // 10, num % 10) + type  # for the "Lytro" dataset
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
                fe_img1 = encoder(img1)
                fe_img2 = encoder(img2)
                fe = torch.max(fe_img1, fe_img2)
                out = decoder(fe, unconsis)

                out_fused = combine(img1, img2, img1_mask_bi, img2_mask_bi, unconsis, out)

                out_fused = out_fused.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).permute(1, 2, 0).cpu()
                out_fused = Image.fromarray(np.uint8(out_fused))
                out_fused.save(save_path + '/MFI-WHU_{}{}_Ours.jpg'.format(num // 10, num % 10))

                toc = time.time()
                print('end Lytro_{}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))
    elif img_nums == 3:
        with torch.no_grad():
            for num in range(1, couples + 1):
                # tic = time.time()
                path1 = file_path + '/lytro-{}{}-A.'.format(num // 10, num % 10) + type  # for the "Lytro" dataset
                path2 = file_path + '/lytro-{}{}-B.'.format(num // 10, num % 10) + type  # for the "Lytro" dataset
                path3 = file_path + '/lytro-{}{}-C.'.format(num // 10, num % 10) + type  # for the "Lytro" dataset
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

                tic = time.time()
                outa = fd(img1, img2, img3)
                out11 = outa[:, 0, :, :].unsqueeze(1)
                out12 = outa[:, 1, :, :].unsqueeze(1)
                confidence_map1 = torch.max(out11, out12)

                outb = fd(img2, img1, img3)
                out21 = outb[:, 0, :, :].unsqueeze(1)
                out22 = outb[:, 1, :, :].unsqueeze(1)
                confidence_map2 = torch.max(out21, out22)

                outc = fd(img3, img1, img2)
                out31 = outc[:, 0, :, :].unsqueeze(1)
                out32 = outc[:, 1, :, :].unsqueeze(1)
                confidence_map3 = torch.max(out31, out32)

                out1 = to_binary(confidence_map1)
                out2 = to_binary(confidence_map2)
                out3 = to_binary(confidence_map3)

                unconsis = find_unconsist(out1, out2, out3)

                fe_img1 = encoder(img1)
                fe_img2 = encoder(img2)
                fe_img3 = encoder(img3)
                fe, _ = torch.max(torch.stack((fe_img1, fe_img2, fe_img3)), dim=0)
                out = decoder(fe, unconsis)
                out_fused = combine(img1, img2, out1, out2, unconsis, out, img3, out3)

                out = out_fused.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).permute(1, 2, 0).cpu()
                out = Image.fromarray(np.uint8(out))
                out.save(save_path + '/Lytro_{}{}.jpg'.format(num // 10, num % 10))

                toc = time.time()
                print('end Lytro_{}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))
    elif img_nums == 4:
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
                img2_read = np.array(img2)
                img3_read = np.array(img3)
                img4_read = np.array(img4)

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
                out11 = outa[:, 0, :, :].unsqueeze(1)
                out12 = outa[:, 1, :, :].unsqueeze(1)
                confidence_map1 = torch.max(out11, out12)

                outb = fd(img2, img1, img3, img4)
                out21 = outb[:, 0, :, :].unsqueeze(1)
                out22 = outb[:, 1, :, :].unsqueeze(1)
                confidence_map2 = torch.max(out21, out22)

                outc = fd(img3, img1, img2, img4)
                out31 = outc[:, 0, :, :].unsqueeze(1)
                out32 = outc[:, 1, :, :].unsqueeze(1)
                confidence_map3 = torch.max(out31, out32)

                outd = fd(img4, img1, img2, img3)
                out41 = outd[:, 0, :, :].unsqueeze(1)
                out42 = outd[:, 1, :, :].unsqueeze(1)
                confidence_map4 = torch.max(out41, out42)

                out1 = to_binary(confidence_map1)
                out2 = to_binary(confidence_map2)
                out3 = to_binary(confidence_map3)
                out4 = to_binary(confidence_map4)

                out1 = fill(out1)
                out2 = fill(out2)
                out3 = fill(out3)
                out4 = fill(out4)

                unconsis = find_unconsist(out1, out2, out3, out4)

                fe_img1 = encoder(img1)
                fe_img2 = encoder(img2)
                fe_img3 = encoder(img3)
                fe_img4 = encoder(img4)
                fe, _ = torch.max(torch.stack((fe_img1, fe_img2, fe_img3, fe_img4)), dim=0)
                out = decoder(fe, unconsis)
                out_fused = combine(img1, img2, out1, out2, unconsis, out, img3, out3, img4, out4)

                out_fused = out_fused.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).permute(1, 2, 0).cpu()
                out_fused = Image.fromarray(np.uint8(out_fused))
                out_fused.save(save_path + '/mffw_{}{}_4.jpg'.format(num // 10, num % 10))

                toc = time.time()
                print('end Lytro_{}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))
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

                unconsis = find_unconsist(out1, out2, out3, out4, out5, out6)

                fe_img1 = encoder(img1)
                fe_img2 = encoder(img2)
                fe_img3 = encoder(img3)
                fe_img4 = encoder(img4)
                fe_img5 = encoder(img5)
                fe_img6 = encoder(img6)
                fe, _ = torch.max(torch.stack((fe_img1, fe_img2, fe_img3, fe_img4, fe_img5, fe_img6)), dim=0)
                out = decoder(fe, unconsis)
                out_fused = combine(img1, img2, out1, out2, unconsis, out, img3, out3, img4, out4,
                                    img5, out5, img6, out6)

                out_fused = out_fused.mul_(255).add_(0.5).clamp_(0, 255).squeeze(0).permute(1, 2, 0).cpu()
                out_fused = Image.fromarray(np.uint8(out_fused))
                out_fused.save(save_path + '/mffw_{}{}_6.jpg'.format(num // 10, num % 10))

                toc = time.time()
                print('end Lytro_{}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))


if __name__ == '__main__':
    # fusion_color('../MFI-WHU', 'jpg', './test_fusion', 30, 2)     # fuse the "MFI-WHU" dataset
    # fusion_color('../Mfif/Lytro_tr', 'jpg', './test_fusion', 20, 2)
    fusion_color('../Mfif/triple_sets', 'jpg', './test_fusion', 9, 3)
    # fusion_color('../triple_sets/four', 'jpg', './test_fusion', 1, 4)
    # fusion_color('../triple_sets/six', 'jpg', './test_fusion', 1, 6)