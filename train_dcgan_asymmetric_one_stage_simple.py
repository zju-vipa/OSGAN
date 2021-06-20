import os
import time
import argparse
import math

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from utils.reader import load_yaml, flatten_dict
from utils.logger import Logger

from utils.registry import get_model, get_optimizer, get_scheduler

from utils.modules import GradientScaler
from utils.nn_tools import get_gradient_ratios

logger = None
cfg = None
num_iter_global = 1

def train_epoch_adv(generator, discriminator, train_loader, optimizers, cfg):
    generator.cuda().train()
    discriminator.cuda().train()

    optimizer_g, optimizer_d = optimizers

    if isinstance(generator, torch.nn.DataParallel):
        zdim = generator.module.zdim
    else:
        zdim = generator.zdim

    global num_iter_global
    loss_d_value = 0.0
    loss_g_value = 0.0

    scaler = GradientScaler.apply
    for i, (x, _) in enumerate(train_loader):
        batch_size = x.size(0)
        x = x.cuda()

        y_real = torch.ones(batch_size).cuda()
        y_fake = torch.zeros(batch_size).cuda()

        z = torch.randn((batch_size, zdim, 1, 1)).cuda()
        fake = generator(z)

        fake_neg = scaler(fake)

        pred_real = discriminator(x)
        pred_fake = discriminator(fake_neg)

        loss_real = F.binary_cross_entropy_with_logits(pred_real, y_real)
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, y_fake, reduction='none')
        loss_d = loss_real + torch.mean(loss_fake)

        # -log(D(G(z))), not included in loss_d, confuse fake image to real
        loss_g = F.binary_cross_entropy_with_logits(pred_fake, y_real, reduction='none')

        gamma = get_gradient_ratios(loss_g, loss_fake, pred_fake)
        
        GradientScaler.factor = gamma

        optimizer_d.zero_grad()
        optimizer_g.zero_grad()

        loss_d.backward()

        optimizer_d.step()
        optimizer_g.step()

        loss_real_val = loss_real.item()
        loss_fake_val = torch.mean(loss_fake).item()
        loss_d_val = loss_d.item()
        loss_g_val = torch.mean(loss_g).item()

        if logger:
            logger.add_scalar('gamma_iter', torch.mean(gamma), num_iter_global)
            logger.add_scalar('loss_g_iter', loss_g_val, num_iter_global)
            logger.add_scalar('loss_d_iter', loss_d_val, num_iter_global)
            logger.add_scalar('loss_real_iter', loss_real_val, num_iter_global)
            logger.add_scalar('loss_fake_iter', loss_fake_val, num_iter_global)

        loss_d_value += loss_d_val * batch_size
        loss_g_value += loss_g_val * batch_size

        num_iter_global += 1

    loss_d_value /= len(train_loader.dataset)
    loss_g_value /= len(train_loader.dataset)

    return loss_d_value, loss_g_value


def train(args=None):
    # ---------------- Configure ----------------
    global cfg
    cfg = load_yaml(args.cfg)
    cfg = dict(cfg)
    global logger
    logger = Logger(log_root='log/'.format(cfg['method'], cfg['dataset']),
                    name="{}-one-stage_{}-asymmetric".format(cfg['method'],
                                                                    cfg['dataset']))

    for k, v in flatten_dict(cfg).items():
        logger.add_text('configuration', "{}: {}".format(k, v))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu_id'])

    # ---------------- Dataset ----------------
    if cfg['dataset'] == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(cfg['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_loader = DataLoader(
            datasets.MNIST(cfg['data_root'], train=True, transform=transform),
            cfg['batch_size'], shuffle=True, num_workers=8)
    elif cfg['dataset'] in ['celeba', 'imagenet', 'FFHQ']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        train_loader = DataLoader(datasets.ImageFolder(cfg['data_root'],
                                                       transform=transform),
                                  cfg['batch_size'], shuffle=True, num_workers=8)
        print('Dataset Size:', len(train_loader.dataset))

    # ---------------- Network ----------------
    generator = get_model('{}_generator'.format(cfg['method']), zdim=cfg['zdim'],
                          num_channel=cfg['num_channel'])
    discriminator = get_model('{}_discriminator'.format(cfg['method']),
                              num_channel=cfg['num_channel'])
    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

    train_epoch = train_epoch_adv

    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    # ---------------- Optimizer ----------------
    optimizer_g = get_optimizer(cfg['optimizer_g']['type'])( \
        generator.parameters(), **cfg['optimizer_g']['args'])
    optimizer_d = get_optimizer(cfg['optimizer_d']['type'])( \
        discriminator.parameters(), **cfg['optimizer_d']['args'])

    scheduler_g, scheduler_d = None, None
    if 'scheduler_g' in cfg.keys():
        scheduler_g = get_scheduler(cfg['scheduler_g']['type'])(optimizer_g,
                                                                **cfg['scheduler_g']['args'])
    if 'scheduler_d' in cfg.keys():
        scheduler_d = get_scheduler(cfg['scheduler_d']['type'])(optimizer_d,
                                                                **cfg['scheduler_d']['args'])

    # ---------------- Training ----------------
    z_fix = torch.randn((100, cfg['zdim'], 1, 1)).cuda()
    if logger:
        dir_save = 'ckpt/{}'.format(logger.log_name)
    else:
        dir_save = 'ckpt/{}-one-stage_{}'.format(cfg['method'], cfg['dataset'])
    os.makedirs(dir_save, exist_ok=True)

    for epoch in range(1, cfg['num_epoch'] + 1):
        loss_d, loss_g = \
            train_epoch(generator, discriminator, train_loader,
                        (optimizer_g, optimizer_d), cfg)

        if scheduler_g: scheduler_g.step()
        if scheduler_d: scheduler_d.step()

        if logger:
            logger.add_scalar('loss_d_epoch', loss_d, epoch)
            logger.add_scalar('loss_g_epoch', loss_g, epoch)

            generator.eval()
            fake_fix = generator(z_fix).cpu()
            fake_fix_pack = make_grid(fake_fix, nrow=16, normalize=True,
                                      range=(-1, 1), pad_value=0.5)
            logger.add_image('fake_fix', fake_fix_pack, epoch)

            zs = torch.randn((100, cfg['zdim'], 1, 1)).cuda()
            fake_rand = generator(zs).cpu()
            fake_rand_pack = make_grid(fake_rand, nrow=16, normalize=True,
                                       range=(-1, 1), pad_value=0.5)
            logger.add_image('fake_rand', fake_rand_pack, epoch)

        torch.save(generator.state_dict(),
                    '{}/generator-epoch{:0>3d}.pth'.format(dir_save, epoch))
                    
        torch.save(discriminator.state_dict(),
                    '{}/discriminator-epoch{:0>3d}.pth'.format(dir_save, epoch))


def parse():
    args = argparse.Namespace()
    # args.cfg = 'cfgs/dcgan/dcgan_asymmetric_one_stage_mnist.yml'
    args.cfg = 'cfgs/dcgan/dcgan_asymmetric_one_stage_celeba_simple.yml'

    return args


if __name__ == '__main__':
    args = parse()
    train(args)
