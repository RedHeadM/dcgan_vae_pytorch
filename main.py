from __future__ import print_function
import argparse
import os
import random
import math
import torch
import itertools
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import logging
import sklearn.utils as sku
from torch.autograd import Variable
from torchtcn.utils.log import log,set_log_file
from torchvision.utils import save_image
from torchtcn.utils.dataset import (DoubleViewPairDataset,
                                    MultiViewVideoDataset, ViewPairDataset)
from torchtcn.utils.comm import get_git_commit_hash,create_dir_if_not_exists
from utils import ReplayBuffer
from torch.nn import functional as F
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
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--saveInt', type=int, default=25, help='number of epochs between checkpoints')
parser.add_argument('--showimg', type=int, default=500, help='number of steps between  image update')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true',default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='/tmp/dc_gan', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
IMAGE_SIZE=(opt.imageSize,opt.imageSize)
log.setLevel(logging.INFO)
set_log_file(os.path.join(opt.outf, "train.log"))
create_dir_if_not_exists(os.path.join(opt.outf, "images"))
log.info("commit hash: {}".format(get_git_commit_hash(__file__)))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
log.info("Random Seed: {}".format( opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    log.warn("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset =='tcn':
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
                                                      number_views=2,
                                                      # std_similar_frame_margin_distribution=sim_frames,
                                                      transform_frames=transformer_train)
    dataset2 = DoubleViewPairDataset(vid_dir=opt.dataroot,
                                                      number_views=2,
                                                      # std_similar_frame_margin_distribution=sim_frames,
                                                      transform_frames=transformer_train)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
#Using itertools.cycle has an important drawback, in that it does not shuffle the data after each iteration:
iter_data=iter(cycle(dataloader2))# WARNING  itertools.cycle  does not shuffle the data after each iteratio

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

    def forward(self,input):
        mu = input[0]
        logvar = input[1]

        std = logvar.mul(0.5).exp_() #calculate the STDEV
        if opt.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_() #random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_() #random normalized noise
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class _Encoder(nn.Module):
    def __init__(self,imageSize):
        super(_Encoder, self).__init__()

        n = math.log2(imageSize)

        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)


        self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv',nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu',nn.LeakyReLU(0.2, inplace=True))
        for i in range(n-3):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid{0}-{1}conv'.format(ngf*2**i, ngf * 2**(i+1)), nn.Conv2d(ngf*2**(i), ngf * 2**(i+1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid{0}batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ngf * 2**(i+1)))
            self.encoder.add_module('pyramid{0}relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*8) x 4 x 4

    def forward(self,input):
        output = self.encoder(input)
        return [self.conv1(output),self.conv2(output)]


class _netG(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(imageSize)
        self.sampler = _Sampler()

        n = math.log2(imageSize)

        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)

        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2**(n-3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2**(n-3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n-3, 0, -1):
            self.decoder.add_module('pyramid{0}-{1}conv'.format(ngf*2**i, ngf * 2**(i-1)),nn.ConvTranspose2d(ngf * 2**i, ngf * 2**(i-1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid{0}batchnorm'.format(ngf * 2**(i-1)), nn.BatchNorm2d(ngf * 2**(i-1)))
            self.decoder.add_module('pyramid{0}relu'.format(ngf * 2**(i-1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False))
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

netG = _netG(opt.imageSize,ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
log.info(str(netG))


class _netD(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        n = math.log2(imageSize)

        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)
        self.main = nn.Sequential()

        # input is (nc) x 64 x 64
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf) x 32 x 32
        for i in range(n-3):
            self.main.add_module('pyramid{0}-{1}conv'.format(ngf*2**(i), ngf * 2**(i+1)), nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i+1), 4, 2, 1, bias=False))
            self.main.add_module('pyramid{0}batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ndf * 2 ** (i+1)))
            self.main.add_module('pyramid{0}relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        # self.main.add_module('output-conv', nn.Conv2d(ndf * 2**(n-3), 1, 4, 1, 0, bias=False))
        self.n_out =1
        self.main.add_module('output-conv', nn.Conv2d(ndf * 2**(n-3), 1*self.n_out, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, self.n_out)


netD = _netD(opt.imageSize,ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
log.info(str(netD))

criterion = nn.BCELoss()
# MSECriterion = nn.MSELoss()
MSECriterion = nn.L1Loss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.make_cuda()
    criterion.cuda()
    MSECriterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# Buffers of previously generated samples
fake_buffer = ReplayBuffer()
gen_win = None
rec_win = None
key_views = ["frames views {}".format(i) for i in range(2)]

d_real_input_noise =0.1

def d_unrolled_loop(batch_size, d_fake_data):
    # 1. Train D on real+fake
    optimizerD.zero_grad()

    #  1A: Train D on real
    d_real_data = next(iter_data) #dataloader.sample(batch_size)#TODO

    if opt.dataset !='tcn':
        d_real_data, _ = d_real_data
    else:
        d_real_data = torch.cat([d_real_data[key_views[0]], d_real_data[key_views[1]]])
    # d_real_data+= torch.randn(d_real_data.data.size()).cuda()*(0.5 *d_real_input_noise)

    if opt.cuda:
        d_real_data = d_real_data.cuda()
    d_real_decision = netD(d_real_data)
    target = torch.ones_like(d_real_decision)
    if opt.cuda:
        target = target.cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    d_fake_decision = netD(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if opt.cuda:
        target = target.cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward(create_graph=True)
    optimizerD.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()


unrolled_steps=10
log.info('unrolled_steps: {}'.format(unrolled_steps))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        # train with real
        if opt.dataset !='tcn':
            real_cpu, _ = data
        else:
            key_views=sku.shuffle(key_views)
            # real_cpu = torch.cat([data[key_views[0]], data[key_views[1]]])
            real_cpu = data[key_views[0]]

        input.data.resize_(real_cpu.size()).copy_(real_cpu)

        label.data.resize_(real_cpu.size(0)).fill_(real_label)
        batch_size = real_cpu.size(0)
        print('batch_size: {}'.format(batch_size))

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        with torch.no_grad():
            gen = netG.decoder(noise)
        gen=fake_buffer.push_and_pop(gen)

        # train real
        input_white_noise = input + torch.randn(input.data.size()).cuda()*(0.5 *d_real_input_noise)
        output = netD(input_white_noise)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        if i % opt.showimg ==0:

            if vis is not None:
                gen_win = vis.image(gen.data[0].cpu()*0.5+0.5,win = gen_win,opts=dict(title='gen fake',width=300,height=300),)
            n=min(batch_size,8)
            imgs= torch.cat([gen[:n]])*0.5+0.5
            save_image(imgs, os.path.expanduser(os.path.join(opt.outf, "images/ep{}_step{}_gen_fake.png".format(epoch,i)))  ,nrow=n)
            save_image(input_white_noise[:n]*0.5+0.5, os.path.expanduser(os.path.join(opt.outf, "images/ep{}_step{}input_white_noise.png".format(epoch,i)))  ,nrow=n)
        label.data.fill_(fake_label)

        output = netD(gen.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        [p.grad.data.clamp_(-5,5) for p in netD.parameters()]
        optimizerD.step()
        ############################
        # (2) Update G network: VAE
        ###########################
        if opt.dataset =='tcn':
            # use the other view
            input.data.resize_(real_cpu.size()).copy_(data[key_views[1]])

        netG.zero_grad()

        encoded = netG.encoder(input)
        mu = encoded[0]
        logvar = encoded[1]

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        sampled = netG.sampler(encoded)
        rec = netG.decoder(sampled)
        if i %opt.showimg==0:
            if vis is not None:
                rec_win = vis.image(rec.data[0].cpu()*0.5+0.5,win = rec_win,opts=dict(title='gen real',width=300,height=300))
            imgs= torch.cat([input[:n],rec[:n]])*0.5+0.5
            save_image(imgs, os.path.expanduser(os.path.join(opt.outf, "images/ep{}_step{}_gen_real.png".format(epoch,i)))  ,nrow=n)
        MSEerr = MSECriterion(rec,input)

        VAEerr = KLD + MSEerr
        VAEerr.backward(retain_graph=True)
        optimizerG.step()

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad() #correct?

        label.data.fill_(real_label)  # fake labels are real for generator cost
#         noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)

        # unroll setp
        if unrolled_steps > 0:
            with torch.no_grad():
                d_fake_data = netG(input)
            backup_D = netD.state_dict()
            backup_optimizerD = optimizerD.state_dict()
            for  _ in range(unrolled_steps):
                d_unrolled_loop(batch_size,d_fake_data)# with real or fake?

        rec = netG(input) # this tensor is freed from mem at this point
        output = netD(rec)
        errG = criterion(output, label)
        errG.backward(retain_graph=True)
        D_G_z2 = output.data.mean()
        [p.grad.data.clamp_(-5,5) for p in netG.decoder.parameters()]
        optimizerG.step()

        if unrolled_steps > 0:
            netD.load_state_dict(backup_D)
            optimizerD.load_state_dict(backup_optimizerD)

            del backup_D,backup_optimizerD


        ###########################
        log.info('[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 VAEerr.data[0], errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

    if epoch%opt.saveInt == 0 and epoch!=0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
