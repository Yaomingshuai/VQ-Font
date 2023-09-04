

import torch

def load_checkpoint(path, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler):
    """
    load_checkpoint
    """
    ckpt = torch.load(path,map_location={'cuda:1': 'cuda:0'})
    gen.load_state_dict(ckpt['generator'])
    g_optim.load_state_dict(ckpt['optimizer'])
    g_scheduler.load_state_dict(ckpt['g_scheduler'])

    if disc is not None:
        disc.load_state_dict(ckpt['discriminator'])
        d_optim.load_state_dict(ckpt['d_optimizer'])
        d_scheduler.load_state_dict(ckpt['d_scheduler'])

    st_epoch = ckpt['epoch'] + 1
    loss = ckpt['loss']

    return st_epoch, loss
