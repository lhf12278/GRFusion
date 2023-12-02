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
from Dataloader import *
from utils import mkdir
from loss import SSIM, TextureLoss
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
parser.add_argument('--root', type=str, default='/your dataset root dir ', help='data path')
parser.add_argument('--save_path', type=str, default='./your save root dir', help='models and pics save path')
parser.add_argument('--save_path_para', type=str, default='./checkpoints', help='models parameter save')
parser.add_argument('--ssl_transformations', type=bool, default=True, help='use ssl_transformations or not')
parser.add_argument('--miniset', type=bool, default=False, help='to choose a mini dataset')
parser.add_argument('--minirate', type=float, default=0.2, help='to detemine the size of a mini dataset')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--epoch', type=int, default=600, help='training epoch')
parser.add_argument('--batch_size', type=int, default=24, help='batchsize')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--summary_name', type=str, default='Fd_', help='Name of the tensorboard summmary')

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
FDModel = U_Net1(3, 2).to(device)
optimizer = optim.SGD([{'params': FDModel.parameters()}],
                        lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam([{'params': FDModel.parameters()}],
                    lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(optimizer, args.epoch)

L1_fun = nn.L1Loss(reduction='mean')
Mse_fun = nn.MSELoss()
Bce_fun = nn.BCELoss()
criterion1 = nn.BCEWithLogitsLoss()
criterion4 = TextureLoss()

# ==================
# Model Training
# ==================
loss_train = []
loss_val = []
mkdir(args.save_path)
print('============ Training Begins ===============')
FDModel.train()

for epoch in tqdm(range(args.epoch)):
    total_loss_per_iter_refresh = 0.
    total_loss_per_epoch_refresh = 0.

    for index, batchdata in enumerate(train_loader, 1):
        ia = batchdata['ia'].to(device)
        ib = batchdata['ib'].to(device)
        ia_mask_label = batchdata['ia_mask_label'].to(device)
        ib_mask_label = batchdata['ib_mask_label'].to(device)

        optimizer.zero_grad()

        out = FDModel(ia, ib)
        out1 = out[:, 0, :, :].unsqueeze(1)
        out2 = out[:, 1, :, :].unsqueeze(1)
        confidence_map = torch.max(out1, out2)
        map3, labela = incrase_C([confidence_map, ia_mask_label])

        img_save = torch.cat([ia, labela], dim=0)

        grid = make_grid(img_save, nrow=ia.shape[0])
        grid1 = grid.mul_(255).add_(0.5).clamp_(0, 255).squeeze().permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        if index % 5 == 0:
            cv2.imwrite(os.path.join(args.save_path, args.summary_name + '_fusion_epoch_' + str(epoch) + '_' + str(
                index) + '_train.png'), grid1)

        ##################### loss function ########################
        ia_loss = Bce_fun(confidence_map, ia_mask_label)

        total_loss_per_iter_refresh = ia_loss

        total_loss_per_iter_refresh.backward()
        optimizer.step()

        total_loss_per_epoch_refresh += total_loss_per_iter_refresh.item()

    print('Epoch:[%d/%d]-----Train------ LOSS:%.4f' % (
        epoch, args.epoch, total_loss_per_epoch_refresh / (len(train_loader))))
    writer.add_scalar('Train/total_task_loss', total_loss_per_epoch_refresh / (len(train_loader)), epoch)

    loss_train.append(total_loss_per_epoch_refresh / (len(train_loader)))
    scheduler.step()

    # ==================
    # Model Validation
    # ==================
    FDModel.eval()
    with torch.no_grad():
        total_loss_per_iter_refresh = 0.
        total_loss_per_epoch_refresh = 0.

        for index, batchdata in enumerate(val_loader):
            ia = batchdata['ia'].to(device)
            ib = batchdata['ib'].to(device)
            ia_mask_label = batchdata['ia_mask_label'].to(device)
            ib_mask_label = batchdata['ib_mask_label'].to(device)

            optimizer.zero_grad()

            out = FDModel(ia, ib)
            out1 = out[:, 0, :, :].unsqueeze(1)
            out2 = out[:, 1, :, :].unsqueeze(1)
            confidence_map = torch.max(out1, out2)

            map3, labela = incrase_C([confidence_map, ia_mask_label])

            ##################### loss function ########################
            ia_loss = Bce_fun(confidence_map, ia_mask_label)
            total_loss_per_iter_refresh = ia_loss
            total_loss_per_epoch_refresh += total_loss_per_iter_refresh.item()

        print('Epoch:[%d/%d]-----Val------ LOSS:%.4f' % (epoch, args.epoch, total_loss_per_epoch_refresh / (len(val_loader))))
        writer.add_scalar('Train/total_task_loss', total_loss_per_epoch_refresh / (len(val_loader)), epoch)
        loss_val.append(total_loss_per_epoch_refresh / (len(val_loader)))

    # ==================
    # Model Saving
    # ==================
    # save models every epoch
    if epoch+1 >= 50:
        torch.save(FDModel.state_dict(),
                   args.save_path_para + '/fd_' + '{}.pth'.format(epoch + 1))

torch.cuda.synchronize()
end = time.time()

# save best models
minloss_index = loss_val.index(min(loss_val))
print("The min loss in validation is obtained in %d epoch" % (minloss_index + 1))
print("The training process has finished! Take a break! ")