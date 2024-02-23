import setproctitle
import torch

setproctitle.setproctitle('Wang Dan')

from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.optim as optim
from BaseModle import *
from FDNet import U_Net1
from FusionNet1 import Encoder, Decoder
from Dataloader import *
from utils import mkdir
import time
import argparse
import log
from tqdm import tqdm
from tensorboardX import SummaryWriter
import cv2

NWORKERS = 2

parser = argparse.ArgumentParser(description='models save and load')
parser.add_argument('--exp_name', type=str, default='Mfif_Fusion', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--root', type=str, default='../Mfif1/data/Train/Fusion_data', help='data path')
parser.add_argument('--save_path', type=str, default='./train_fusion', help='models and pics save path')
parser.add_argument('--save_path_para', type=str, default='./checkpoints_fusion', help='models parameter save')
parser.add_argument('--ssl_transformations', type=bool, default=True, help='use ssl_transformations or not')
parser.add_argument('--miniset', type=bool, default=False, help='to choose a mini dataset')
parser.add_argument('--minirate', type=float, default=0.2, help='to detemine the size of a mini dataset')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--epoch', type=int, default=500, help='training epoch')
parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--summary_name', type=str, default='Fusion_',
                    help='Name of the tensorboard summmary')
parser.add_argument('--fddetect', type=str, default="./checkpoints/kernel_37.pth",
                    help='the dir of test results to save')

args = parser.parse_args()
writer = SummaryWriter(comment=args.summary_name)

# ==================
# init
# ==================
io = log.IOStream(args)
io.cprint(str(args))
toPIL = transforms.ToPILImage()
np.random.seed(1)  # to get the same images and leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# Read Data
# ==================
train_augmentation = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(256),
                                                     torchvision.transforms.RandomHorizontalFlip()
                                                     ])

dataset = Fusion_data(io, args, args.root, transform=None, gray=True, partition='train')

# Creating data indices for training and validation splits:
train_indices = dataset.train_ind
val_indices = dataset.val_ind

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)  # sampler will assign the whole data according to batchsize.
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                          sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, num_workers=NWORKERS, batch_size=args.batch_size,
                        sampler=valid_sampler)

torch.cuda.synchronize()
start = time.time()

# ==================
# Init Model
# ==================
FDmodel = U_Net1(3, 2).to(device)
FDmodel.load_state_dict(torch.load(args.fddetect))
Decoder1 = FDmodel.cuda()
Decoder1.eval()

Encoder = Encoder().to(device)
Decoder = Decoder().to(device)

optimizer = optim.SGD([{'params': Encoder.parameters()}, {'params': Decoder.parameters()}],
                        lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam([{'params': Encoder.parameters()}, {'params': Decoder.parameters()}],
                    lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(optimizer, args.epoch)

L1_fun = nn.L1Loss(reduction='mean')

# ==================
# Model Training
# ==================
loss_train = []
loss_val = []
mkdir(args.save_path)
print('============ Training Begins ===============')
Encoder.train()
Decoder.train()

for epoch in tqdm(range(args.epoch)):
    total_loss_per_iter_refresh = 0.
    total_loss_per_epoch_refresh = 0.

    for index, batchdata in enumerate(train_loader, 1):
        ia = batchdata['ia'].to(device)
        ib = batchdata['ib'].to(device)
        fuse_label = batchdata['fuse_label'].to(device)

        out1 = FDmodel(ia, ib)
        out11 = out1[:, 0, :, :].unsqueeze(1)
        out12 = out1[:, 1, :, :].unsqueeze(1)
        confidence_map1 = torch.max(out11, out12)

        out2 = FDmodel(ia, ib)
        out21 = out2[:, 0, :, :].unsqueeze(1)
        out22 = out2[:, 1, :, :].unsqueeze(1)
        confidence_map2 = torch.max(out21, out22)

        ia_mask_bi = to_binary(confidence_map1)
        ib_mask_bi = to_binary(confidence_map2)
        unconsis = find_unconsist(ia_mask_bi, ib_mask_bi)

        optimizer.zero_grad()
        fe_ia = Encoder(ia)
        fe_ib = Encoder(ib)

        fe = torch.max(fe_ia, fe_ib)
        out = Decoder(fe, unconsis)

        out_fused = combine(ia, ib, ia_mask_bi, ib_mask_bi, unconsis, out)

        ###################### display the output ##################
        img_save = torch.cat([ia, ib, out, out_fused, fuse_label], dim=0)

        grid = make_grid(img_save, nrow=ia.shape[0])
        grid1 = grid.mul_(255).add_(0.5).clamp_(0, 255).squeeze().permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        if index % 150 == 0:
            cv2.imwrite(os.path.join(args.save_path, args.summary_name + '_fusion_epoch_' + str(epoch) + '_' + str(
                index) + '_train.png'), grid1)

        ##################### loss function ########################
        out_l1_loss = L1_fun(out, fuse_label)
        mar_l1_loss = L1_fun(out * unconsis, fuse_label * unconsis)
        l1_loss = out_l1_loss + 0.05 * mar_l1_loss

        total_loss_per_iter_refresh = l1_loss

        total_loss_per_iter_refresh.backward()
        optimizer.step()

        total_loss_per_epoch_refresh += total_loss_per_iter_refresh

    print('Epoch:[%d/%d]-----Train------ LOSS:%.4f' % (
        epoch, args.epoch, total_loss_per_epoch_refresh / (len(train_loader))))
    writer.add_scalar('Train/total_task_loss', total_loss_per_epoch_refresh / (len(train_loader)), epoch)

    loss_train.append(total_loss_per_epoch_refresh / (len(train_loader)))
    scheduler.step()

    # ==================
    # Model Validation
    # ==================
    Encoder.eval()
    Decoder.eval()
    with torch.no_grad():
        total_loss_per_iter_refresh = 0.
        total_loss_per_epoch_refresh = 0.

        for index, batchdata in enumerate(val_loader):
            ia = batchdata['ia'].to(device)
            ib = batchdata['ib'].to(device)
            fuse_label = batchdata['fuse_label'].to(device)

            out1 = FDmodel(ia, ib)
            out11 = out1[:, 0, :, :].unsqueeze(1)
            out12 = out1[:, 1, :, :].unsqueeze(1)
            confidence_map1 = torch.max(out11, out12)

            out2 = FDmodel(ia, ib)
            out21 = out2[:, 0, :, :].unsqueeze(1)
            out22 = out2[:, 1, :, :].unsqueeze(1)
            confidence_map2 = torch.max(out21, out22)

            ia_mask_bi = to_binary(confidence_map1)
            ib_mask_bi = to_binary(confidence_map1)

            optimizer.zero_grad()
            fe_ia = Encoder(ia)
            fe_ib = Encoder(ib)
            fe = torch.max(fe_ia, fe_ib)
            out = Decoder(fe, unconsis)
            # out = Decoder(fe)

            out_fused = combine(ia, ib, ia_mask_bi, ib_mask_bi, unconsis, out)

            ##################### loss function ########################
            out_l1_loss = L1_fun(out, fuse_label)
            mar_l1_loss = L1_fun(out * unconsis, fuse_label * unconsis)
            l1_loss = out_l1_loss + 0.05*mar_l1_loss
            total_loss_per_iter_refresh = l1_loss

            total_loss_per_epoch_refresh += total_loss_per_iter_refresh

        print('Epoch:[%d/%d]-----Val------ LOSS:%.4f' % (epoch, args.epoch, total_loss_per_epoch_refresh / (len(val_loader))))
        writer.add_scalar('Train/total_task_loss', total_loss_per_epoch_refresh / (len(val_loader)), epoch)
        loss_val.append(total_loss_per_epoch_refresh / (len(val_loader)))

    # ==================
    # Model Saving
    # ==================
    if epoch > 100:
        torch.save(Encoder.state_dict(),
                   args.save_path_para + '/Encoder_' + '{}.pth'.format(epoch + 1))
        torch.save(Decoder.state_dict(),
                   args.save_path_para + '/Decoder_' + '{}.pth'.format(epoch + 1))
torch.cuda.synchronize()
end = time.time()

# save best models
minloss_index = loss_val.index(min(loss_val))
print("The min loss in validation is obtained in %d epoch" % (minloss_index + 1))
print("The training process has finished! Take a break! ")