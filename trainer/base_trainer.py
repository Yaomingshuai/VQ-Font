import copy
from tqdm import trange
from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from .trainer_utils import *
from pathlib import Path
try:
    from apex import amp
except ImportError:
    print('failed to import apex')
from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

class BaseTrainer:
    """
    BaseTrainer
    """
    def __init__(self,ddp_gpu, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg):

        self.ddp_gpu = ddp_gpu
        self.gen = gen
        self.gen_ema = copy.deepcopy(self.gen)
        self.g_optim = g_optim
        self.g_scheduler = g_scheduler
        self.disc = disc
        self.d_optim = d_optim
        self.d_scheduler = d_scheduler
        self.cfg = cfg
        self.perceptual_loss = LPIPS().eval().cuda()
        

        [self.gen, self.gen_ema, self.disc], [self.g_optim, self.d_optim] = self.set_model(
            [self.gen, self.gen_ema, self.disc],
            [self.g_optim, self.d_optim]
        )

        self.logger = logger
        self.evaluator = evaluator
        self.cv_loaders = cv_loaders

        self.step = 1

        self.g_losses = {}
        self.d_losses = {}


    def set_model(self, models, opts):
        return models, opts


    def clear_losses(self):
        """ Integrate & clear loss json_dict """
        # g losses
        loss_dic = {k: v.item() for k, v in self.g_losses.items()}
        loss_dic['g_total'] = sum(loss_dic.values())
        # d losses
        loss_dic.update({k: v.item() for k, v in self.d_losses.items()})
        # ac losses
        self.g_losses = {}
        self.d_losses = {}

        return loss_dic

    def accum_g(self, decay=0.9):
        """
        ema
        """
        # with paddle.no_grad():
        # with torch.no_grad():
        #     param_dict_src = dict(self.gen.named_parameters())
        #     for p_name, p_tgt in self.gen_ema.named_parameters():
        #         p_src = param_dict_src[p_name]
        #         assert(p_src is not p_tgt)
        #         p_tgt.set_value(decay*p_tgt + (1. - decay)*p_src)
        par1 = dict(self.gen_ema.named_parameters())
        par2 = dict(self.gen.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))

    def sync_g_ema(self, in_style_ids, in_comp_ids, in_imgs, trg_style_ids, trg_comp_ids,
                   content_imgs):
        return

    def train(self):
        return
    
    def add_l1_loss_only_mainstructure(self, out, target):
        """
        add_pixel_loss
        """
        loss1 = F.l1_loss(out, target, reduction="mean")
        self.g_losses['l1'] = loss1*2
              
        return loss1 
    
    def add_lpips_loss_only_mainstructure(self, out, target):
        
        p_loss = self.perceptual_loss(out, target)
        self.g_losses['lpips'] = torch.mean(p_loss)
        
        return torch.mean(p_loss)

    def add_l1_loss(self, out, target,self_infer):
        """
        add_pixel_loss
        """
        loss1 = F.l1_loss(out, target, reduction="mean") 
        loss2 = F.l1_loss(self_infer, target, reduction="mean")*0.5
        
        self.g_losses['l1'] = loss1+loss2
              
        return loss1+loss2
    
    def add_lpips_loss(self, out, target,self_infer):
        
        p_loss1 = self.perceptual_loss(out, target)
        p_loss2 = self.perceptual_loss(self_infer, target)*0.5
        self.g_losses['lpips'] = torch.mean(p_loss1)+torch.mean(p_loss2)
        
        return torch.mean(p_loss1)+torch.mean(p_loss2)
        
    
    def add_pixel_loss(self, out, target,self_infer):
        """
        add_pixel_loss
        """
        loss1 = F.mse_loss(out, target, reduction="mean") 
        loss2 = F.mse_loss(self_infer, target, reduction="mean") * 0.5
        self.g_losses['pixel'] = loss1+loss2
        
        return loss1
        # return loss1
    
    def add_cosin_loss(self, out, target,self_infer):
        """
        add_pixel_loss
        """
        out = out.permute(0,2,3,1).contiguous().view(-1,256)
        target = target.permute(0,2,3,1).contiguous().view(-1,256)
        self_infer = self_infer.permute(0,2,3,1).contiguous().view(-1,256)
        loss = torch.nn.CosineEmbeddingLoss()
        loss1 = loss(self_infer,target,torch.tensor([1]).cuda())
        loss2 = loss(out,target,torch.tensor([1]).cuda())
        self.g_losses['cosin'] = loss1 + loss2 
        # self.g_losses['pixel'] = loss1 
        
        return loss1 + loss2
    
    def add_crossentropy_loss(self, out, target,self_infer):

        loss1 = F.cross_entropy(out,target)
        loss2 = F.cross_entropy(self_infer,target)*0.5
        self.g_losses['cross'] = (loss1 + loss2)
        # self.g_losses['pixel'] = loss1 
        
        return loss1 + loss2
    
    def add_feat_loss(self, out, target):

        loss1 = F.smooth_l1_loss(out,target.detach())*0.01
        
        self.g_losses['feat'] = loss1
        # self.g_losses['pixel'] = loss1 
        
        return loss1
    
    def add_crossentropy_loss_only_mainstructure(self, out, target):
        """
        add_pixel_loss
        """
        loss1 = F.cross_entropy(out,target)
        self.g_losses['cross'] = loss1 
        # self.g_losses['pixel'] = loss1 
        
        return loss1
        
    
    def add_vq_loss(self,z_e_x, z_q_x, z_e_x_self,z_q_x_self):
        """
        add_pixel_loss
        """
        loss1 = F.mse_loss(z_e_x, z_q_x.detach()) 
        loss2 = F.mse_loss(z_e_x_self, z_q_x_self.detach())
        
        self.g_losses['vq'] = 0.25*(loss1 + loss2)
        
        return loss1 + loss2

    def add_gan_g_loss(self, real_font, real_uni, fake_uni, fake_stru):
        """
        add_gan_g_loss
        """
        if self.cfg['gan_w'] == 0.:
            return 0.

        # g_loss = -(fake_uni.mean() + fake_stru.mean())
        g_loss = -( fake_uni.mean())
        g_loss *= self.cfg['gan_w']
        self.g_losses['gen'] = g_loss*0.002

        return g_loss
    
    def add_gan_g_loss_vqgan(self, logit_fake):
        """
        add_gan_g_loss
        """
        if self.cfg['gan_w'] == 0.:
            return 0.

        g_loss = -(logit_fake.mean())
        g_loss *= self.cfg['gan_w']
        self.g_losses['gen'] = g_loss*0.5

        return g_loss
    
    def add_gan_d_loss(self, real_stru, real_uni, fake_stru, fake_uni):
        """
        add_gan_d_loss
        """
        if self.cfg['gan_w'] == 0.:
            return 0.
        
        # d_loss = (F.relu(1. - real_stru).mean() + F.relu(1. + fake_stru).mean()) + \
        #          F.relu(1. - real_uni).mean() + F.relu(1. + fake_uni).mean()
        
        d_loss = F.relu(1. - real_uni).mean() + F.relu(1. + fake_uni).mean()
        
        d_loss *= self.cfg['gan_w']
        self.d_losses['disc'] = d_loss*0.002

        return d_loss
    
    def add_gan_d_loss_vqgan(self, logit_fake, logit_real):
        """
        add_gan_d_loss
        """
        if self.cfg['gan_w'] == 0.:
            return 0.
        
        d_loss = F.relu(1. - logit_real).mean() + F.relu(1. + logit_fake).mean()
        
        d_loss *= self.cfg['gan_w']
        self.d_losses['disc'] = d_loss*0.5

        return d_loss

    def d_backward(self):
        """
        d_backward
        """
        with utils.temporary_freeze(self.gen):
            d_loss = sum(self.d_losses.values())
            d_loss.backward()

    def g_backward(self):
        """
        g_backward
        """
        with utils.temporary_freeze(self.disc):
            g_loss = sum(self.g_losses.values())
            g_loss.backward()


    def save(self, cur_loss, method, save_freq=None):
        """
        Args:
            method: all / last
                all: save checkpoint by step
                last: save checkpoint to 'last.pdparams'
                all-last: save checkpoint by step per save_freq and
                          save checkpoint to 'last.pdparams' always
        """
        if method not in ['all', 'last', 'all-last']:
            return

        step_save = False
        last_save = False
        if method == 'all' or (method == 'all-last' and self.step % save_freq == 0):
            step_save = True
        if method == 'last' or method == 'all-last':
            last_save = True
        assert step_save or last_save

        save_dic = {
            'generator': self.gen.state_dict(),
            'generator_ema': self.gen_ema.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'optimizer': self.g_optim.state_dict(),
            'epoch': self.step,
            'loss': cur_loss
        }
        if self.disc is not None:
            save_dic['discriminator'] = self.disc.state_dict()
            save_dic['d_optimizer'] = self.d_optim.state_dict()
            save_dic['d_scheduler'] = self.d_scheduler.state_dict()


        ckpt_dir = self.cfg['work_dir'] / "checkpoints" / self.cfg['unique_name']
        step_ckpt_name = "{:06d}-{}.pdparams".format(self.step, self.cfg['name'])
        last_ckpt_name = "last.pdparams"
        step_ckpt_path = Path.cwd() /ckpt_dir / step_ckpt_name
        last_ckpt_path = ckpt_dir / last_ckpt_name

        log = ""
        if step_save:
            torch.save(save_dic, str(step_ckpt_path))
            log = "Checkpoint is saved to {}".format(step_ckpt_path)

            if last_save:
                utils.rm(last_ckpt_path)
                last_ckpt_path.symlink_to(step_ckpt_path)
                log += " and symlink to {}".format(last_ckpt_path)

        if not step_save and last_save:
            utils.rm(last_ckpt_path)  # last
            torch.save(save_dic, str(last_ckpt_path))
            log = "Checkpoint is saved to {}".format(last_ckpt_path)

        self.logger.info("{}\n".format(log))

    def baseplot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val
        }

        if self.disc is not None:
            tag_scalar_dic.update({
                'train/d_loss': losses.disc.val,
                'train/g_loss': losses.gen.val,
                'train/d_real_font': discs.real_font.val,
                'train/d_real_uni': discs.real_uni.val,
                'train/d_fake_font': discs.fake_font.val,
                'train/d_fake_uni': discs.fake_uni.val,
            })


    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
                .format(step=self.step, L=losses, D=discs, S=stats))
