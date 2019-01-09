from __future__ import print_function

import argparse
import itertools
import logging
import math
import os
import random

import numpy as np

import sklearn.utils as sku
import torch
import torch.backends.cudnn as cudnn
import torch.legacy.nn as lnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from sklearn import preprocessing
from torch.autograd import Variable
from torch.nn import functional as F
from torchtcn.utils.comm import create_dir_if_not_exists, get_git_commit_hash
from torchtcn.utils.dataset import (DoubleViewPairDataset,
                                    MultiViewVideoDataset, ViewPairDataset)
from torchtcn.utils.log import log, set_log_file
from torchvision.utils import save_image
from utils import ReplayBuffer, create_lable_func

try:
    import visdom
    vis = visdom.Visdom()
    vis.env = 'vae_dcgan'
except (ImportError, AttributeError):
    vis = None
    print("visdom not used")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=1000, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1e6, help='number of epochs to train for')
parser.add_argument('--saveInt', type=int, default=10000,
                    help='number of epochs between checkpoints')
parser.add_argument('--showimg', type=int, default=500,
                    help='number of steps between  image update')

#
parser.add_argument('--g_lr', type=float, default=0.0001)
parser.add_argument('--d_lr', type=float, default=0.0004)
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.9)

parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='/tmp/dc_gan',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
IMAGE_SIZE = (opt.imageSize, opt.imageSize)
log.setLevel(logging.INFO)
set_log_file(os.path.join(opt.outf, "train.log"))
create_dir_if_not_exists(os.path.join(opt.outf, "images"))
log.info("commit hash: {}".format(get_git_commit_hash(__file__)))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
log.info("Random Seed: {}".format(opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    log.warn("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset == 'tcn':
    num_views = 2
    # opt.batchSize=opt.batchSize//2 #num views
    transformer_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE[0]),  # TODO reize here Resize()
        # transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    sampler = None
    shuffle = True
    # only one view pair in batch
    # sim_frames = 5
    dataset = DoubleViewPairDataset(vid_dir=opt.dataroot,
                                    add_camera_info=True,
                                    number_views=num_views,
                                    # std_similar_frame_margin_distribution=sim_frames,
                                    transform_frames=transformer_train)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, pin_memory=opt.cuda,
                                         shuffle=True, num_workers=int(opt.workers))


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# Using itertools.cycle has an important drawback, in that it does not shuffle the data after each iteration:
# WARNING  itertools.cycle  does not shuffle the data after each iteratio
iter_data = iter(cycle(dataloader))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()

    def forward(self, input):
        mu = input[0]
        logvar = input[1]

        std = logvar.mul(0.5).exp_()  # calculate the STDEV
        if opt.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class _Encoder(nn.Module):
    def __init__(self, imageSize):
        super(_Encoder, self).__init__()

        n = math.log2(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))
        for i in range(n-3):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid{0}-{1}conv'.format(ngf*2**i, ngf * 2**(i+1)),
                                    nn.Conv2d(ngf*2**(i), ngf * 2**(i+1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid{0}batchnorm'.format(
                ngf * 2**(i+1)), nn.BatchNorm2d(ngf * 2**(i+1)))
            self.encoder.add_module('pyramid{0}relu'.format(
                ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*8) x 4 x 4

    def forward(self, input):
        output = self.encoder(input)
        return [self.conv1(output), self.conv2(output)]


class _netG(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(imageSize)
        self.sampler = _Sampler()

        n = math.log2(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz,
                                                                 ngf * 2**(n-3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2**(n-3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n-3, 0, -1):
            self.decoder.add_module('pyramid{0}-{1}conv'.format(ngf*2**i, ngf * 2**(i-1)),
                                    nn.ConvTranspose2d(ngf * 2**i, ngf * 2**(i-1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid{0}batchnorm'.format(
                ngf * 2**(i-1)), nn.BatchNorm2d(ngf * 2**(i-1)))
            self.decoder.add_module('pyramid{0}relu'.format(
                ngf * 2**(i-1)), nn.LeakyReLU(0.2, inplace=True))
            # self.decoder.add_module('pyramid{0}dropout'.format(ngf * 2**(i-1)), nn.Dropout(p=0.5))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf,      nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.sampler, output, range(self.ngpu))
            output = nn.parallel.data_parallel(self.decoder, output, range(self.ngpu))
        else:
            output = self.encoder(input)
            output = self.sampler(output)
            output = self.decoder(output)
        return output

    def make_cuda(self):
        self.encoder.cuda()
        self.sampler.cuda()
        self.decoder.cuda()


class _netD(nn.Module):
    def __init__(self, imageSize, ngpu, n_lables=[]):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        n = math.log2(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)
        self.main = nn.Sequential()

        # input is (nc) x 64 x 64
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf) x 32 x 32
        for i in range(n-3):
            self.main.add_module('pyramid{0}-{1}conv'.format(ngf*2**(i), ngf * 2**(i+1)),
                                 nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i+1), 4, 2, 1, bias=False))
            self.main.add_module('pyramid{0}batchnorm'.format(
                ngf * 2**(i+1)), nn.BatchNorm2d(ndf * 2 ** (i+1)))
            self.main.add_module('pyramid{0}relu'.format(
                ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        # self.main.add_module('output-conv', nn.Conv2d(ndf * 2**(n-3), 1, 4, 1, 0, bias=False))

        self.n_lables = n_lables

        self.cov_out_fake_real = nn.Conv2d(ndf * 2**(n-3), 1, 4, 1, 0, bias=False)

        self.cov_out_labels = nn.ModuleList()
        for i, l in enumerate(n_lables):
            self.cov_out_labels.append(nn.Conv2d(ndf * 2**(n-3), l, 4, 1, 0, bias=False))

        self.fakeout_sigmoid = nn.Sigmoid()
#         if self.n_lables:
        # self.lable_softmax=nn.Softmax()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        labels_out = []
        for lout, nout_l in zip(self.cov_out_labels, self.n_lables):
            labels_out.append(lout(output).view(-1, nout_l))
        out_fake_real = self.fakeout_sigmoid(self.cov_out_fake_real(output)).view(-1, 1)
        return out_fake_real, labels_out


unrolled_steps = 0
log.info('unrolled_steps: {}'.format(unrolled_steps))
use_lables = True
log.info('use_lables: {}'.format(use_lables))
lable_cnt = 0

if opt.dataset == 'tcn' and use_lables:
    max_len_vid = max(max(l) for l in dataset.frame_lengths)
    mean_len_mean = np.mean(np.mean(dataset.frame_lengths))
    # add a lable for n frames
    lable_n_frames = 20
    lable_frame_look_up = []
    for i in range(max_len_vid):
        if i % lable_n_frames == 0 and i != 0 and i < mean_len_mean:
            lable_cnt += 1
        lable_frame_look_up.append(lable_cnt)
    lable_cnt = np.max(lable_cnt)+1
    c_criterion = nn.CrossEntropyLoss().cuda()
    log.info('lable_cnt: {}'.format(lable_cnt))
    assert lable_cnt


criterion_GAN = nn.BCELoss().cuda()
criterion_id = nn.L1Loss().cuda()
criterion_lables = nn.CrossEntropyLoss().cuda()


fake_buffer = ReplayBuffer()
gen_win = None
rec_win = None
key_views = ["frames views {}".format(i) for i in range(num_views)]

lable_keys_cam_view_info = []  # list with keys for view 0 and view 1

#
for view_i in range(num_views):
    lable_keys_cam_view_info.append(["cam_pitch_view_{}".format(view_i),
                                     "cam_yaw_view_{}".format(view_i),
                                     "cam_distance_view_{}".format(view_i)])
mapping_cam_info_lable = {}
mapping_cam_info_one_hot = {}
n_bins = 10
# create a different mapping for echt setting
for cam_info_view in lable_keys_cam_view_info:
    for cam_inf in cam_info_view:
        if "pitch" in cam_inf:
            min_val, max_val = 0, 10
        elif "yaw" in cam_inf:
            min_val, max_val = 0, 10
        elif "distance" in cam_inf:
            min_val, max_val = 0, 10
        to_l, to_hot_l = create_lable_func(min_val, max_val, n_bins)
        mapping_cam_info_lable[cam_inf] = to_l
        mapping_cam_info_one_hot[cam_inf] = to_hot_l

assert opt.dataset == 'tcn'


# create the nets
netG = _netG(opt.imageSize, ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
log.info(str(netG))

netD = _netD(opt.imageSize, ngpu, n_lables=[n_bins]*len(mapping_cam_info_lable))
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
log.info(str(netD))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, opt.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2))
# Buffers of previously generated samples
netD.cuda()
netG.cuda()


def get_cam_labels(data):
    lables = []
    if lable_cnt:
        lables_c = [lable_frame_look_up[frame] for frame in data["frame index"].numpy()]
        lables_c = torch.tensor(lables_c, dtype=torch.long).cuda()
        lables_c_fake = torch.tensor(np.random.randint(
            0, lable_cnt, batch_size), dtype=torch.long).cuda()

    # pitch roll, dist
    return cam_0_one_hot, cam_1_one_hot


def kl_loss(encoded):
    mu, logvar = encoded
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    return torch.sum(KLD_element).mul_(-0.5)


def save_2imgs(imgs, t, step, n):
    imgs = torch.cat([imgs[0][:n], imgs[1][:n]])*0.5+0.5
    save_image(imgs, os.path.expanduser(os.path.join(
        opt.outf, "images/step{}_{}.png".format(step, t))), nrow=n)


# Loss weights
lambda_cyc = 10.
lambda_id = 0.5 * lambda_cyc

for step in range(int(opt.niter)):
    data = next(iter_data)
    #####################
    # CYCLE loss train G and decoder
    ######################
    optimizerG.zero_grad()

    key_views = sku.shuffle(key_views)
    # real_cpu = torch.cat([data[key_views[0]], data[key_views[1]]])
    real_view_0 = Variable(data[key_views[0]], requires_grad=False).cuda()
    real_view_1 = Variable(data[key_views[1]], requires_grad=False).cuda()
    # r/f lables: Adversarial ground truths
    batch_size = real_view_0.size(0)
    label_valid = Variable(torch.Tensor(np.ones((batch_size,))), requires_grad=False).cuda()
    label_fake = Variable(torch.Tensor(np.zeros((batch_size,))), requires_grad=False).cuda()
    print("real_view_0", real_view_0.type())
    encoded_view_0 = netG.encoder(real_view_0)
    # TODO add cam as input for decoder
    decoded_fake_view_1 = netG.decoder(netG.sampler(encoded_view_0))
    encoded_view_1 = netG.encoder(decoded_fake_view_1)
    # back to view 0
    decoded_fake_view_0 = netG.decoder(netG.sampler(encoded_view_1))
    # loss
    kl = (kl_loss(encoded_view_0)+kl_loss(encoded_view_1))/2.
    id_loss = (criterion_id(decoded_fake_view_0, real_view_0) +
               criterion_id(decoded_fake_view_1, real_view_1))/2.
    gan_loss = (criterion_GAN(netD(decoded_fake_view_0)[0], label_valid)
                + criterion_GAN(netD(decoded_fake_view_1)[0], label_valid))/2.

    loss = lambda_cyc * gan_loss + lambda_id * (kl+id_loss)

    if step % opt.showimg == 0:
        save_2imgs([real_view_0, decoded_fake_view_0], "view0", step, 8)
        save_2imgs([real_view_1, decoded_fake_view_1], "view1", step, 8)
    loss.backward()
    optimizerG.step()

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    optimizerD.zero_grad()

    # Real loss
    d_out, lables = netD(torch.cat([real_view_0, real_view_1]))
    loss_real = criterion_GAN(d_out, torch.cat([label_valid, label_valid]))
    # real lable loss
    label_loss = 0
    # for l in lab TODO2
    # Fake loss (on batch of previously generated samples)
    fake_img = torch.cat([decoded_fake_view_0, decoded_fake_view_1])
    fake_img = fake_buffer.push_and_pop(fake_img)
    loss_fake = criterion_GAN(netD(fake_img.detach())[0], torch.cat([label_fake, label_fake]))

    # Total loss
    loss_D = (loss_real + loss_fake) / 2

    loss_D.backward()
    optimizerD.step()
    #
    # if step % opt.saveInt == 0 and step != 0:
    #     torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, step))
    #     torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, step))
