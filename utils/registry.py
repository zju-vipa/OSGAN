import torch
import torch.optim as optim

from torchvision import datasets, transforms as T

import networks

MODEL_DICT = {
    'dcgan_generator': networks.dcgan.DCGenerator,
    'dcgan_discriminator': networks.dcgan.DCDiscriminator,

    'dcgan_gn_generator': networks.dcgan.DCGenerator,
    'dcgan_gn_discriminator': networks.dcgan.DCDiscriminatorGN,

    'wgan_dcgan_generator': networks.dcgan.DCGenerator,
    'wgan_dcgan_discriminator': networks.dcgan.DCDiscriminator,

    'wgan_dcgan_gn_generator': networks.dcgan.DCGenerator,
    'wgan_dcgan_gn_discriminator': networks.dcgan.DCDiscriminatorGN,
}


def get_model(name: str, **kwargs):
    return MODEL_DICT[name](**kwargs)


def get_optimizer(name):
    name = name.lower()
    if name == 'sgd':
        return optim.SGD
    elif name == 'adam':
        return optim.Adam
    elif name == 'rmsprop':
        return optim.rmsprop


def get_scheduler(name):
    name = name.lower()
    if name == 'multisteplr':
        return optim.lr_scheduler.MultiStepLR
    elif name == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR
