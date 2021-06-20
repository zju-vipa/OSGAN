import os
import time
import argparse

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from utils.reader import load_yaml, flatten_dict
from utils.logger import Logger
from utils.nn_tools import GradientHook
from utils.registry import get_model, get_optimizer, get_scheduler

logger = None
cfg = None
num_iter_global = 1


def train_epoch_adv(generator, discriminator, train_loader, optimizers):
    generator.cuda().train()
    discriminator.cuda().train()

    optimizer_g, optimizer_d = optimizers

    if isinstance(generator, torch.nn.DataParallel):
        zdim = generator.module.zdim
    else:
        zdim = generator.zdim

    global num_iter_global
    loss_adv_value = 0.0
    for x, _ in train_loader:
        batch_size = x.size(0)
        x = x.cuda()

        # Note: the target for wgan is different from vanilla gan
        y_real = torch.ones(batch_size).neg_().cuda()
        y_fake = torch.ones(batch_size).cuda()
        target = torch.cat((y_real, y_fake), dim=0)

        z = torch.randn((batch_size, zdim, 1, 1)).cuda()
        fake = generator(z)

        data = torch.cat((x, fake), dim=0)

        pred = discriminator(data)

        loss_adv = torch.mean(pred * target)

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        loss_adv.backward()

        optimizer_d.step()
        optimizer_g.step()

        # ---------- weight clipping for Lipschitz constraint ----------
        for param in discriminator.parameters():
            param.data.clamp_(-0.01, 0.01)

        if logger:
            logger.add_scalar('loss_adv_iter', loss_adv.item(), num_iter_global)

        loss_adv_value += loss_adv.item() * batch_size

        num_iter_global += 1

    loss_adv_value /= len(train_loader.dataset)

    return loss_adv_value, None


def train(args=None):
    # ---------------- Configure ----------------
    global cfg
    cfg = load_yaml(args.cfg)
    cfg = dict(cfg)
    global logger
    logger = Logger(log_root='log/'.format(cfg['method'], cfg['dataset']),
                    name="{}-one-stage_{}".format(cfg['method'],
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

    # ---------------- Set Hook Func ----------------
    hook_grad = GradientHook(generator)
    hook_grad.set_negate_grads_hook()

    train_epoch = train_epoch_adv


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
        loss_adv, loss_bn = \
            train_epoch(generator, discriminator, train_loader,
                        (optimizer_g, optimizer_d))

        if scheduler_g: scheduler_g.step()
        if scheduler_d: scheduler_d.step()

        if logger:
            logger.add_scalar('loss_adv_epoch', loss_adv, epoch)
            if loss_bn:
                logger.add_scalar('loss_bn_epoch', loss_bn, epoch)

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

        if isinstance(generator, torch.nn.DataParallel):
            torch.save(generator.module.state_dict(),
                       '{}/generator-epoch{:0>3d}.pth'.format(dir_save, epoch))
        else:
            torch.save(generator.state_dict(),
                       '{}/generator-epoch{:0>3d}.pth'.format(dir_save, epoch))
        if isinstance(discriminator, torch.nn.DataParallel):
            torch.save(discriminator.module.state_dict(),
                       '{}/discriminator-epoch{:0>3d}.pth'.format(dir_save, epoch))
        else:
            torch.save(discriminator.state_dict(),
                       '{}/discriminator-epoch{:0>3d}.pth'.format(dir_save, epoch))


def parse():
    args = argparse.Namespace()
    args.cfg = 'cfgs/wgan/wgan_dcgan_one_stage_celeba.yml'
    return args


if __name__ == '__main__':
    args = parse()
    train(args)
