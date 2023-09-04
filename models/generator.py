import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from omegaconf import OmegaConf
from .comp_encoder import comp_enc_builder
from .content_encoder import content_enc_builder
from .decoder import Integrator
from .memory import Memory
from functools import partial
from .former import TransformerSALayer



def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Generator(nn.Module):
    """
    Generator
    """
    def __init__(self, C_in, C, C_out, cfg, comp_enc, dec, content_enc):
        super().__init__()
        configs = [OmegaConf.load(cfg) for cfg in ['vqgan/custom_vqgan.yaml']]
        config = OmegaConf.merge(*configs, {})
        model = instantiate_from_config(config.model)
        model.init_from_ckpt('vqgan/1024_16*16_vaecoder.ckpt')
        self.vqgan = model  
        for name,para in self.vqgan.named_parameters():
            print(name)
            if name not in ['decoder.layers.0.conv1.conv.weight','decoder.layers.0.conv1.conv.bias','decoder.layers.0.conv2.conv.weight', \
                'decoder.layers.0.conv2.conv.bias','decoder.layers.1.conv1.conv.weight','decoder.layers.1.conv1.conv.bias', \
                    'decoder.layers.1.conv2.conv.weight','decoder.layers.1.conv2.conv.bias',\
                        'decoder.layers.2.conv1.conv.weight','decoder.layers.2.conv1.conv.bias', \
                    'decoder.layers.2.conv2.conv.weight','decoder.layers.2.conv2.conv.bias',\
                   'post_quant_conv.weight','post_quant_conv.bias']:
            
                para.requires_grad_(True)

        self.former = nn.Sequential(*[TransformerSALayer(embed_dim=256, nhead=8, dim_mlp=512, dropout=0.0) 
                                    for _ in range(15)])
        self.position_emb = nn.Parameter(torch.zeros(256, 256))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1024)
        )
        self.component_encoder = comp_enc_builder(C_in, C, **comp_enc)

        #memory
        self.memory = Memory()
        self.num_heads = cfg.num_heads
        self.shot = cfg.kshot
        num_channels = 256
        
        self.linears_key = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_value = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_query = nn.Linear(num_channels, num_channels, bias=False)
        self.fc = nn.Linear(num_channels, num_channels, bias=False)
        self.layer_norm = nn.LayerNorm(num_channels,  eps=1e-6)
        C_content = content_enc['C_out']#  C_out: 256
        self.content_encoder = content_enc_builder(
            C_in, C, **content_enc                 
        )#content_final:256 16 16    
        self.atten_emb = nn.Sequential(nn.Embedding(25,256),nn.Sigmoid())
        self.comp_emb = nn.Sequential(nn.Embedding(25,256),nn.Sigmoid())
        self.token_emb = nn.Sequential(nn.Embedding(25,256),nn.Sigmoid())
        self.fuse_emb = nn.Sequential(nn.Embedding(25,256),nn.Sigmoid())
        self.integrator = partial(Integrator, norm='in', activ='relu', weight_init='xavier')(C*8, C_content=C_content)
    
    def reset_memory(self):
        """
        reset_memory
        """
        self.memory.reset_memory()

    def get_kqv_matrix(self, fm, linears):
        #matmul with style featuremaps and content featuremaps
        ret = linears(fm)
        return ret

    def encode_write_comb(self, style_ids, style_sample_index, style_imgs, style_imgs_crose,style_imgs_fine,in_stru_ids,reset_memory=True):
        """
        encode_write_comb
        """
        if reset_memory:
            self.reset_memory()

        feat_scs= self.component_encoder(style_imgs)
        self.memory.write_comb(style_ids, style_sample_index, feat_scs['last'])

        return feat_scs

    def read_memory(self, target_style_ids, trg_sample_index, reset_memory=True,reduction='mean'):
        """
        read_memory
        """   
        feats = self.memory.read_chars(target_style_ids, trg_sample_index, reduction=reduction)
        feats = torch.stack([x for x in feats]) #[B,3,C,H,W]
        batch, shot, channel, h, w = feats.shape
        feats = torch.transpose(feats, 2,3)
        feats = torch.transpose(feats, 3,4) #B,3HW,C 
        feats_reshape = torch.reshape(feats, (batch, -1, channel)) #先只用最后一层做transformer B,3HW,C 32,768,256
        ######### attention ########
        d_channel = int(channel / self.num_heads)#size: [B, 3HW, num_heads, C/num_head]
        key_matrix = self.get_kqv_matrix(feats_reshape, self.linears_key)
        key_matrix = key_matrix
        key_matrix = torch.reshape(key_matrix, (batch, -1, self.num_heads, d_channel))
        key_matrix = torch.transpose(key_matrix,1,2) #[B, num_heads, 3HW, C/num_heads]
        value_matrix = self.get_kqv_matrix(feats_reshape, self.linears_value)
        value_matrix = torch.reshape(value_matrix, (batch, -1, self.num_heads, d_channel))
        value_matrix = torch.transpose(value_matrix, 1,2)

        if reset_memory:
            self.reset_memory()

        return key_matrix, value_matrix

    def cont_similarity(self,tar_stru,similarity):
        sim =[]
        if tar_stru==0:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:7,:])
                b = torch.mean(i_[7:,:])
                sim.append(a)
                sim.append(b)
        if tar_stru==1:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:,:])
                sim.append(a)
        if tar_stru==2:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[0:5,:])
                c = torch.mean(i_[5:8,:])
                b = torch.mean(i_[8:,:])
                sim.append(a)
                sim.append(b)
                sim.append(c)
        if tar_stru==3:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_)
                sim.append(a)
        if tar_stru==4:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:,0:7])
                b = torch.mean(i_[:,7:])
                sim.append(a)
                sim.append(b)
        if tar_stru==5:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:,:])
                sim.append(a)
        if tar_stru==6:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:,:])
                sim.append(a)
        if tar_stru==7:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:,:])
                sim.append(a)
        if tar_stru==8:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:-3,8:-3])
                b = torch.mean(i_[:-3,:8])*(104.0/152.0)+torch.mean(i_[-3:,:])*(48.0/152.0)
                sim.append(a)
                sim.append(b)
        if tar_stru==9:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[6:,5:])
                b = torch.mean(i_[:6,:])*(96.0/146.0)+torch.mean(i_[6:,:5])*(50.0/146.0)
                sim.append(a)
                sim.append(b)
        if tar_stru==10:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:,:6])
                b = torch.mean(i_[:,7:11])
                c = torch.mean(i_[:,11:])
                sim.append(a)
                sim.append(b)
                sim.append(c)                
        if tar_stru==11:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:,:])
                sim.append(a)
        if tar_stru==12:
            for i_ in similarity:
                i_.squeeze(2)
                a = torch.mean(i_[:,:])
                sim.append(a)
        return sim
        
    def refer_similarity(self,in_stru,refer):
        res=[]
        num=[]
        for index,i in enumerate(in_stru):
            if i == 0:
                a = torch.mean(refer[index][:,:,:,:7,:],dim=(3,4))
                b = torch.mean(refer[index][:,:,:,7:,:],dim=(3,4))
                res.append(a)
                res.append(b)
                num.append(2)               
            if i == 1:
                a = torch.mean(refer[index][:,:,:,:,:],dim=(3,4))
                res.append(a)
                num.append(1)                
            if i == 2:
                a = torch.mean(refer[index][:,:,:,0:5,:],dim=(3,4))
                c = torch.mean(refer[index][:,:,:,5:8,:],dim=(3,4))
                b = torch.mean(refer[index][:,:,:,8:,:],dim=(3,4))
                res.append(a)
                res.append(b)
                res.append(c)
                num.append(3)
            if i == 3:
                a = torch.mean(refer[index][:,:,:,:,:],dim=(3,4))
                res.append(a)
                num.append(1)               
            if i == 4:
                a = torch.mean(refer[index][:,:,:,:,:7],dim=(3,4))
                b = torch.mean(refer[index][:,:,:,:,7:],dim=(3,4))
                res.append(a)
                res.append(b)
                num.append(2)                
            if i == 5:
                a = torch.mean(refer[index][:,:,:,:,:],dim=(3,4))
                res.append(a)
                num.append(1)                
            if i == 6:
                a = torch.mean(refer[index][:,:,:,:,:],dim=(3,4))
                res.append(a)
                num.append(1)                
            if i == 7:
                a = torch.mean(refer[index][:,:,:,:,:],dim=(3,4))
                res.append(a)
                num.append(1)                
            if i == 8:
                a = torch.mean(refer[index][:,:,:,:-3,8:],dim=(3,4))
                b = torch.mean(refer[index][:,:,:,:-3,:8],dim=(3,4))*(104.0/152.0)+torch.mean(refer[index][:,:,:,-3:,:],dim=(3,4))*(48.0/152.0)
                res.append(a)
                res.append(b)
                num.append(2)                
            if i == 9:
                a = torch.mean(refer[index][:,:,:,6:,5:],dim=(3,4))
                b = torch.mean(refer[index][:,:,:,:6,:],dim=(3,4))*(96.0/146.0)+torch.mean(refer[index][:,:,:,6:,:5],dim=(3,4))*(50.0/146.0)
                res.append(a)
                res.append(b)
                num.append(2)                
            if i == 10:
                a = torch.mean(refer[index][:,:,:,:,:6],dim=(3,4))
                b = torch.mean(refer[index][:,:,:,:,7:11],dim=(3,4))
                c = torch.mean(refer[index][:,:,:,:,11:],dim=(3,4))
                res.append(a)
                res.append(b)
                res.append(c)
                num.append(3)                                
            if i == 11:
                a = torch.mean(refer[index][:,:,:,:,:],dim=(3,4))
                res.append(a)
                num.append(1)                                                
            if i == 12:
                a = torch.mean(refer[index][:,:,:,:,:],dim=(3,4))
                res.append(a)
                num.append(1)                                                
        return res ,num           

    def fusion_am(self,refer,in_stru,tar_stru,similarity):
        atten_map = []
        flag=0
        refer_res = []
        for index, i in enumerate(in_stru):
            refer_ = refer[index].clone()
            if tar_stru==0 :      
                if i==0 :         
                    refer_[:,:7,:,0,:7,:] = refer[index][:,:7,:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:7,:] = refer[index][:,7:,:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:7,:,0,7:,:] = refer[index][:,:7,:,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:7,:] = refer[index][:,7:,:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==1 or i ==3 or i==5 or i==6 or i==7 or i==11 or i==12:          
                    refer_[:,:7,:,0,:,:] = refer[index][:,:7,:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:,:] = refer[index][:,7:,:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==2 :  
                    refer_[:,:7,:,0,0:5,:] = refer[index][:,:7,:,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,0:5,:] = refer[index][:,7:,:,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:7,:,0,8:,:] = refer[index][:,:7,:,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,8:,:] = refer[index][:,7:,:,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:7,:,0,5:8,:] = refer[index][:,:7,:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,5:8,:] = refer[index][:,7:,:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)      
                if i==4 :
                    refer_[:,:7,:,0,:,:7] = refer[index][:,:7,:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:,:7] = refer[index][:,7:,:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:7,:,0,:,7:] = refer[index][:,:7,:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:,7:] = refer[index][:,7:,:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==8 :
                    refer_[:,:7,:,0,:-3,8:] = refer[index][:,:7,:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:-3,8:] = refer[index][:,7:,:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:7,:,0,:-3,:8] = refer[index][:,:7,:,0,:-3,:8]+similarity[flag]
                    refer_[:,:7,:,0,-3:,:] = refer[index][:,:7,:,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:-3,:8] = refer[index][:,7:,:,0,:-3,:8]+similarity[flag]
                    refer_[:,7:,:,0,-3:,:] = refer[index][:,7:,:,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==9 :
                    refer_[:,:7,:,0,6:,5:] = refer[index][:,:7,:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,6:,5:] = refer[index][:,7:,:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:7,:,0,6:,:5] = refer[index][:,:7,:,0,6:,:5]+similarity[flag]
                    refer_[:,:7,:,0,:6,:] = refer[index][:,:7,:,0,:6,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,6:,:5] = refer[index][:,7:,:,0,6:,:5]+similarity[flag]
                    refer_[:,7:,:,0,:6,:] = refer[index][:,7:,:,0,:6,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==10 :
                    refer_[:,:7,:,0,:,:6] = refer[index][:,:7,:,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:,:6] = refer[index][:,7:,:,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,:7,:,0,:,7:11] = refer[index][:,:7,:,0,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:,7:11] = refer[index][:,7:,:,0,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,:7,:,0,:,11:] = refer[index][:,:7,:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:,11:] = refer[index][:,7:,:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_) 
 
            if tar_stru==1 or tar_stru==3 or tar_stru==5 or tar_stru==6 or tar_stru==7 or tar_stru==11 or tar_stru==12:      
                if i==0 :         
                    refer_[:,:,:,0,:7,:] = refer[index][:,:,:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:,0,7:,:] = refer[index][:,:,:,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==1 or i ==3 or i==5 or i==6 or i==7 or i==11 or i==12:          
                    refer_[:,:,:,0,:,:] = refer[index][:,:,:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==2 :  
                    refer_[:,:,:,0,0:5,:] = refer[index][:,:,:,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:,0,8:,:] = refer[index][:,:,:,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:,0,5:8,:] = refer[index][:,:,:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)      
                if i==4 :
                    refer_[:,:,:,0,:,:7] = refer[index][:,:,:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:,0,:,7:] = refer[index][:,:,:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==8 :
                    refer_[:,:,:,0,:-3,8:] = refer[index][:,:,:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:,0,:-3,:8] = refer[index][:,:,:,0,:-3,:8]+similarity[flag]
                    refer_[:,:,:,0,-3:,:] = refer[index][:,:,:,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==9 :
                    refer_[:,:,:,0,6:,5:] = refer[index][:,:,:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:,0,6:,:5] = refer[index][:,:,:,0,6:,:5]+similarity[flag]
                    refer_[:,:,:,0,:6,:] = refer[index][:,:,:,0,:6,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==10 :
                    refer_[:,:,:,0,:,:6] = refer[index][:,:,:,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:,0,:,7:11] = refer[index][:,:,:,0,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:,0,:,11:] = refer[index][:,:,:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_) 
            
            if tar_stru==2 :      
                if i==0 :         
                    refer_[:,0:5,:,0,:7,:] = refer[index][:,0:5,:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:7,:] = refer[index][:,8:,:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:7,:] = refer[index][:,5:8,:,0,:7,:] +similarity[flag]
                    flag=flag+1
                    refer_[:,0:5,:,0,7:,:] = refer[index][:,0:5,:,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,7:,:] = refer[index][:,8:,:,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,7:,:] = refer[index][:,5:8,:,0,7:,:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==1 or i ==3 or i==5 or i==6 or i==7 or i==11 or i==12:          
                    refer_[:,0:5,:,0,:,:] = refer[index][:,0:5,:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:,:] = refer[index][:,8:,:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:,:] = refer[index][:,5:8,:,0,:,:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==2 :  
                    refer_[:,0:5,:,0,0:5,:] = refer[index][:,0:5,:,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,0:5,:] = refer[index][:,8:,:,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,0:5,:] = refer[index][:,5:8,:,0,0:5,:] +similarity[flag]
                    flag=flag+1
                    refer_[:,0:5,:,0,8:,:] = refer[index][:,0:5,:,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,8:,:] = refer[index][:,8:,:,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,8:,:] = refer[index][:,5:8,:,0,8:,:] +similarity[flag]
                    flag=flag+1
                    refer_[:,0:5,:,0,5:8,:] = refer[index][:,0:5,:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,5:8,:] = refer[index][:,8:,:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,5:8,:] = refer[index][:,5:8,:,0,5:8,:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)      
                if i==4 :
                    refer_[:,0:5,:,0,:,:7] = refer[index][:,0:5,:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:,:7] = refer[index][:,8:,:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:,:7] = refer[index][:,5:8,:,0,:,:7] +similarity[flag]
                    flag=flag+1
                    refer_[:,0:5,:,0,:,7:] = refer[index][:,0:5,:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:,7:] = refer[index][:,8:,:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:,7:] = refer[index][:,5:8,:,0,:,7:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==8 :
                    refer_[:,0:5,:,0,:-3,8:] = refer[index][:,0:5,:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:-3,8:] = refer[index][:,8:,:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:-3,8:] = refer[index][:,5:8,:,0,:-3,8:] +similarity[flag]
                    flag=flag+1                
                    refer_[:,0:5,:,0,:-3,:8] = refer[index][:,0:5,:,0,:-3,:8]+similarity[flag]
                    refer_[:,0:5,:,0,:-3:,:] = refer[index][:,0:5,:,0,:-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:-3,:8] = refer[index][:,8:,:,0,:-3,:8] +similarity[flag]
                    refer_[:,8:,:,0,:-3:,:] = refer[index][:,8:,:,0,:-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:-3,:8] = refer[index][:,5:8,:,0,:-3,:8] +similarity[flag]
                    refer_[:,5:8,:,0,:-3:,:] = refer[index][:,5:8,:,0,:-3:,:] +similarity[flag]
                    flag=flag+1                   
                    refer_res.append(refer_)
                if i==9 :
                    refer_[:,0:5,:,0,6:,5:] = refer[index][:,0:5,:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,6:,5:] = refer[index][:,8:,:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,6:,5:] = refer[index][:,5:8,:,0,6:,5:] +similarity[flag]
                    flag=flag+1                
                    refer_[:,0:5,:,0,:6,:] = refer[index][:,0:5,:,0,:6,:]+similarity[flag]
                    refer_[:,0:5,:,0,6:,:5] = refer[index][:,0:5,:,0,6:,:5]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:6,:] = refer[index][:,8:,:,0,:6,:] +similarity[flag]
                    refer_[:,8:,:,0,6:,:5] = refer[index][:,8:,:,0,6:,:5]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:6,:] = refer[index][:,5:8,:,0,:6,:] +similarity[flag]
                    refer_[:,5:8,:,0,6:,:5] = refer[index][:,5:8,:,0,6:,:5] +similarity[flag]
                    flag=flag+1   
                    refer_res.append(refer_)
                if i==10 :
                    refer_[:,0:5,:,0,:,:6] = refer[index][:,0:5,:,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:,:6] = refer[index][:,8:,:,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:,:6] = refer[index][:,5:8,:,0,:,:6] +similarity[flag]
                    flag=flag+1
                    refer_[:,0:5,:,0,:,7:11] = refer[index][:,0:5,:,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:,7:11] = refer[index][:,8:,:,0,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:,7:11] = refer[index][:,5:8,:,0,:,7:11] +similarity[flag]
                    flag=flag+1
                    refer_[:,0:5,:,0,:,11:] = refer[index][:,0:5,:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_[:,8:,:,0,:,11:] = refer[index][:,8:,:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_[:,5:8,:,0,:,11:] = refer[index][:,5:8,:,0,:,11:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_) 
            
            if tar_stru==4 :      
                if i==0 :         
                    refer_[:,:,:7,0,:7,:] = refer[index][:,:,:7,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:7,:] = refer[index][:,:,7:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:7,0,7:,:] = refer[index][:,:,:7,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:7,:] = refer[index][:,:,7:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==1 or i ==3 or i==5 or i==6 or i==7 or i==11 or i==12:          
                    refer_[:,:,:7,0,:,:] = refer[index][:,:,:7,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:,:] = refer[index][:,:,7:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==2 :  
                    refer_[:,:,:7,0,0:5,:] = refer[index][:,:,:7,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,0:5,:] = refer[index][:,:,7:,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:7,0,8:,:] = refer[index][:,:,:7,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,8:,:] = refer[index][:,:,7:,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:7,0,5:8,:] = refer[index][:,:,:7,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,5:8,:] = refer[index][:,:,7:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)      
                if i==4 :
                    refer_[:,:,:7,0,:,:7] = refer[index][:,:,:7,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:,:7] = refer[index][:,:,7:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:7,0,:,7:] = refer[index][:,:,:7,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:,7:] = refer[index][:,:,7:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==8 :
                    refer_[:,:,:7,0,:-3,8:] = refer[index][:,:,:7,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:-3,8:] = refer[index][:,:,7:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:7,0,:-3,:8] = refer[index][:,:,:7,0,:-3,:8]+similarity[flag]
                    refer_[:,:,:7,0,-3::,:] = refer[index][:,:,:7,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:-3,:8] = refer[index][:,:,7:,0,:-3,:8]+similarity[flag]
                    refer_[:,:,7:,0,-3:,:] = refer[index][:,:,7:,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==9 :
                    refer_[:,:,:7,0,6:,5:] = refer[index][:,:,:7,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,6:,5:] = refer[index][:,:,7:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:7,0,6:,:5] = refer[index][:,:,:7,0,6:,:5]+similarity[flag]
                    refer_[:,:,:7,0,:6,:] = refer[index][:,:,:7,0,:6,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,6:,:5] = refer[index][:,:,7:,0,6:,:5]+similarity[flag]
                    refer_[:,:,7:,0,:6,:] = refer[index][:,:,7:,0,:6,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==10 :
                    refer_[:,:,:7,0,:,:6] = refer[index][:,:,:7,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:,:6] = refer[index][:,:,7:,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:7,0,:,7:11] = refer[index][:,:,:7,0,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:,0,:,7:11] = refer[index][:,:,7:,0,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,:7,0,:,11:] = refer[index][:,:,:7,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_[:,7:,:,0,:,11:] = refer[index][:,7:,:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_) 

            if tar_stru==8 :      
                if i==0 :         
                    refer_[:,:-3,8:,0,:7,:] = refer[index][:,:-3,8:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:7,:] = refer[index][:,:-3,:8,0,:7,:]+similarity[flag]
                    refer_[:,-3:,:,0,:7,:] = refer[index][:,-3:,:,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,8:,0,7:,:] = refer[index][:,:-3,8:,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,7:,:] = refer[index][:,:-3,:8,0,7:,:]+similarity[flag]
                    refer_[:,-3:,:,0,7:,:] = refer[index][:,-3:,:,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==1 or i ==3 or i==5 or i==6 or i==7 or i==11 or i==12:          
                    refer_[:,:-3,8:,0,:,:] = refer[index][:,:-3,8:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:,:] = refer[index][:,:-3,:8,0,:,:]+similarity[flag]
                    refer_[:,-3:,:,0,:,:] = refer[index][:,-3:,:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==2 :  
                    refer_[:,:-3,8:,0,0:5,:] = refer[index][:,:-3,8:,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,0:5,:] = refer[index][:,:-3,:8,0,0:5,:]+similarity[flag]
                    refer_[:,-3:,:,0,0:5,:] = refer[index][:,-3:,:,0,0:5,:]+similarity[flag]
                    flag=flag+1                    
                    refer_[:,:-3,8:,0,8:,:] = refer[index][:,:-3,8:,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,8:,:] = refer[index][:,:-3,:8,0,8:,:]+similarity[flag]
                    refer_[:,-3:,:,0,8:,:] = refer[index][:,-3:,:,0,8:,:]+similarity[flag]
                    flag=flag+1                   
                    refer_[:,:-3,8:,0,5:8,:] = refer[index][:,:-3,8:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,5:8,:] = refer[index][:,:-3,:8,0,5:8,:]+similarity[flag]
                    refer_[:,-3:,:,0,5:8,:] = refer[index][:,-3:,:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)      
                if i==4 :
                    refer_[:,:-3,8:,0,:,:7] = refer[index][:,:-3,8:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:,:7] = refer[index][:,:-3,:8,0,:,:7]+similarity[flag]
                    refer_[:,-3:,:,0,:,:7] = refer[index][:,-3:,:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,8:,0,:,7:] = refer[index][:,:-3,8:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:,7:] = refer[index][:,:-3,:8,0,:,7:]+similarity[flag]
                    refer_[:,-3:,:,0,:,7:] = refer[index][:,-3:,:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==8 :
                    refer_[:,:-3,8:,0,:-3,8:] = refer[index][:,:-3,8:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:-3,8:] = refer[index][:,:-3,:8,0,:-3,8:]+similarity[flag]
                    refer_[:,-3:,:,0,:-3,8:] = refer[index][:,-3:,:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,8:,0,:-3,:8] = refer[index][:,:-3,8:,0,:-3,:8]+similarity[flag]
                    refer_[:,:-3,8:,0,-3:,:] = refer[index][:,:-3,8:,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:-3,:8] = refer[index][:,:-3,:8,0,:-3,:8]+similarity[flag]
                    refer_[:,:-3,:8,0,-3:,:] = refer[index][:,:-3,:8,0,-3:,:]+similarity[flag]
                    refer_[:,-3:,:,0,:-3,:8] = refer[index][:,-3:,:,0,:-3,:8]+similarity[flag]
                    refer_[:,-3:,:,0,-3:,:] = refer[index][:,-3:,:,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==9 :
                    refer_[:,:-3,8:,0,6:,5:] = refer[index][:,:-3,8:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,6:,5:] = refer[index][:,:-3,:8,0,6:,5:]+similarity[flag]
                    refer_[:,-3:,:,0,6:,5:] = refer[index][:,-3:,:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,8:,0,:6,:] = refer[index][:,:-3,8:,0,:6,:]+similarity[flag]
                    refer_[:,:-3,8:,0,6:,:5] = refer[index][:,:-3,8:,0,6:,:5]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:6,:] = refer[index][:,:-3,:8,0,:6,:]+similarity[flag]
                    refer_[:,:-3,:8,0,6:,:5] = refer[index][:,:-3,:8,0,6:,:5]+similarity[flag]
                    refer_[:,-3:,:,0,:6,:] = refer[index][:,-3:,:,0,:6,:]+similarity[flag]
                    refer_[:,-3:,:,0,6:,:5] = refer[index][:,-3:,:,0,6:,:5]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==10 :
                    refer_[:,:-3,8:,0,:,:6] = refer[index][:,:-3,8:,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:,:6] = refer[index][:,:-3,:8,0,:,:6]+similarity[flag]
                    refer_[:,-3:,:,0,:,:6] = refer[index][:,-3:,:,0,:,:6]+similarity[flag]
                    flag=flag+1                    
                    refer_[:,:-3,8:,0,7:11] = refer[index][:,:-3,8:,0,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,7:11] = refer[index][:,:-3,:8,0,7:11]+similarity[flag]
                    refer_[:,-3:,:,0,7:11] = refer[index][:,-3:,:,0,7:11]+similarity[flag]
                    flag=flag+1                   
                    refer_[:,:-3,8:,0,:,11:] = refer[index][:,:-3,8:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:-3,:8,0,:,11:] = refer[index][:,:-3,:8,0,:,11:]+similarity[flag]
                    refer_[:,-3:,:,0,:,11:] = refer[index][:,-3:,:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_) 
            
            if tar_stru==9 :      
                if i==0 :         
                    refer_[:,6:,5:,0,:7,:] = refer[index][:,6:,5:,0,:7,:] +similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:7,:] = refer[index][:,:6,:,0,:7,:]+similarity[flag]
                    refer_[:,6:,:5,0,:7,:] = refer[index][:,6:,:5,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,6:,5:,0,7:,:] = refer[index][:,6:,5:,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,7:,:] = refer[index][:,:6,:,0,7:,:]+similarity[flag]
                    refer_[:,6:,:5,0,7:,:] = refer[index][:,6:,:5,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==1 or i ==3 or i==5 or i==6 or i==7 or i==11 or i==12:          
                    refer_[:,6:,5:,0,:,:] = refer[index][:,6:,5:,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:,:] = refer[index][:,:6,:,0,:,:]+similarity[flag]
                    refer_[:,6:,:5,0,:,:] = refer[index][:,6:,:5,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==2 :  
                    refer_[:,6:,5:,0,:5,:] = refer[index][:,6:,5:,0,:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:5,:] = refer[index][:,:6,:,0,:5,:]+similarity[flag]
                    refer_[:,6:,:5,0,:5,:] = refer[index][:,6:,:5,0,:5,:]+similarity[flag]
                    flag=flag+1                    
                    refer_[:,6:,5:,0,8:,:] = refer[index][:,6:,5:,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,8:,:] = refer[index][:,:6,:,0,8:,:]+similarity[flag]
                    refer_[:,6:,:5,0,8:,:] = refer[index][:,6:,:5,0,8:,:]+similarity[flag]
                    flag=flag+1                   
                    refer_[:,6:,5:,0,5:8,:] = refer[index][:,6:,5:,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,5:8,:] = refer[index][:,:6,:,0,5:8,:]+similarity[flag]
                    refer_[:,6:,:5,0,5:8,:] = refer[index][:,6:,:5,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)      
                if i==4 :
                    refer_[:,6:,5:,0,:,:7] = refer[index][:,6:,5:,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:,:7] = refer[index][:,:6,:,0,:,:7]+similarity[flag]
                    refer_[:,6:,:5,0,:,:7] = refer[index][:,6:,:5,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,6:,5:,0,:,7:] = refer[index][:,6:,5:,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:,7:] = refer[index][:,:6,:,0,:,7:]+similarity[flag]
                    refer_[:,6:,:5,0,:,7:] = refer[index][:,6:,:5,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==8 :
                    refer_[:,6:,5:,0,:-3,8:] = refer[index][:,6:,5:,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:-3,8:] = refer[index][:,:6,:,0,:-3,8:]+similarity[flag]
                    refer_[:,6:,:5,0,:-3,8:] = refer[index][:,6:,:5,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,6:,5:,0,:-3,:8] = refer[index][:,6:,5:,0,:-3,:8]+similarity[flag]
                    refer_[:,6:,5:,0,-3:,:] = refer[index][:,6:,5:,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:-3,:8] = refer[index][:,:6,:,0,:-3,:8]+similarity[flag]
                    refer_[:,:6,:,0,-3:,:] = refer[index][:,:6,:,0,-3:,:]+similarity[flag]
                    refer_[:,6:,:5,0,:-3,:8] = refer[index][:,6:,:5,0,:-3,:8]+similarity[flag]
                    refer_[:,6:,:5,0,-3:,:] = refer[index][:,6:,:5,0,-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==9 :
                    refer_[:,6:,5:,0,6:,5:] = refer[index][:,6:,5:,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,6:,5:] = refer[index][:,:6,:,0,6:,5:]+similarity[flag]
                    refer_[:,6:,:5,0,6:,5:] = refer[index][:,6:,:5,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,6:,5:,0,:6,:] = refer[index][:,6:,5:,0,:6,:]+similarity[flag]
                    refer_[:,6:,5:,0,6:,:5] = refer[index][:,6:,5:,0,6:,:5]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:6,:] = refer[index][:,:6,:,0,:6,:]+similarity[flag]
                    refer_[:,:6,:,0,6:,:5] = refer[index][:,:6,:,0,6:,:5]+similarity[flag]
                    refer_[:,6:,:5,0,:6,:] = refer[index][:,6:,:5,0,:6,:]+similarity[flag]
                    refer_[:,6:,:5,0,6:,:5] = refer[index][:,6:,:5,0,6:,:5]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==10 :
                    refer_[:,6:,5:,0,:,:6] = refer[index][:,6:,5:,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:,:6] = refer[index][:,:6,:,0,:,:6]+similarity[flag]
                    refer_[:,6:,:5,0,:,:6] = refer[index][:,6:,:5,0,:,:6]+similarity[flag]
                    flag=flag+1                    
                    refer_[:,6:,5:,0,7:11] = refer[index][:,6:,5:,0,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,7:11] = refer[index][:,:6,:,0,7:11]+similarity[flag]
                    refer_[:,6:,:5,0,7:11] = refer[index][:,6:,:5,0,7:11]+similarity[flag]
                    flag=flag+1                   
                    refer_[:,6:,5:,0,:,11:] = refer[index][:,6:,5:,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:6,:,0,:,11:] = refer[index][:,:6,:,0,:,11:]+similarity[flag]
                    refer_[:,6:,:5,0,:,11:] = refer[index][:,6:,:5,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_) 
               
            if tar_stru==10 :      
                if i==0 :         
                    refer_[:,:,:6,0,:7,:] = refer[index][:,:,:6,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:7,:] = refer[index][:,:,7:11,0,:7,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:7,:] = refer[index][:,:,11:,0,:7,:] +similarity[flag]
                    flag=flag+1
                    refer_[:,:,:6,0,7:,:] = refer[index][:,:,:6,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,7:,:] = refer[index][:,:,7:11,0,7:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,7:,:] = refer[index][:,:,11:,0,7:,:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==1 or i ==3 or i==5 or i==6 or i==7 or i==11 or i==12:          
                    refer_[:,:,:6,0,:,:] = refer[index][:,:,:6,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:,:] = refer[index][:,:,7:11,0,:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:,:] = refer[index][:,:,11:,0,:,:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==2 :  
                    refer_[:,:,:6,0,0:5,:] = refer[index][:,:,:6,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,0:5,:] = refer[index][:,:,7:11,0,0:5,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,0:5,:] = refer[index][:,:,11:,0,0:5,:] +similarity[flag]
                    flag=flag+1
                    refer_[:,:,:6,0,8:,:] = refer[index][:,:,:6,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,8:,:] = refer[index][:,:,7:11,0,8:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,8:,:] = refer[index][:,:,11:,0,8:,:] +similarity[flag]
                    flag=flag+1
                    refer_[:,:,:6,0,5:8,:] = refer[index][:,:,:6,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,5:8,:] = refer[index][:,:,7:11,0,5:8,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,5:8,:] = refer[index][:,:,11:,0,5:8,:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)      
                if i==4 :
                    refer_[:,:,:6,0,:,:7] = refer[index][:,:,:6,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:,:7] = refer[index][:,:,7:11,0,:,:7]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:,:7] = refer[index][:,:,11:,0,:,:7] +similarity[flag]
                    flag=flag+1
                    refer_[:,:,:6,0,:,7:] = refer[index][:,:,:6,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:,7:] = refer[index][:,:,7:11,0,:,7:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:,7:] = refer[index][:,:,11:,0,:,7:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_)
                if i==8 :
                    refer_[:,:,:6,0,:-3,8:] = refer[index][:,:,:6,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:-3,8:] = refer[index][:,:,7:11,0,:-3,8:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:-3,8:] = refer[index][:,:,11:,0,:-3,8:] +similarity[flag]
                    flag=flag+1                
                    refer_[:,:,:6,0,:-3,:8] = refer[index][:,:,:6,0,:-3,:8]+similarity[flag]
                    refer_[:,:,:6,0,:-3:,:] = refer[index][:,:,:6,0,:-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:-3,:8] = refer[index][:,:,7:11,0,:-3,:8] +similarity[flag]
                    refer_[:,:,7:11,0,:-3:,:] = refer[index][:,:,7:11,0,:-3:,:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:-3,:8] = refer[index][:,:,11:,0,:-3,:8] +similarity[flag]
                    refer_[:,:,11:,0,:-3:,:] = refer[index][:,:,11:,0,:-3:,:] +similarity[flag]
                    flag=flag+1                   
                    refer_res.append(refer_)
                if i==9 :
                    refer_[:,:,:6,0,6:,5:] = refer[index][:,:,:6,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,6:,5:] = refer[index][:,:,7:11,0,6:,5:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,6:,5:] = refer[index][:,:,11:,0,6:,5:] +similarity[flag]
                    flag=flag+1                
                    refer_[:,:,:6,0,:6,:] = refer[index][:,:,:6,0,:6,:]+similarity[flag]
                    refer_[:,:,:6,0,6:,:5] = refer[index][:,:,:6,0,6:,:5]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:6,:] = refer[index][:,:,7:11,0,:6,:] +similarity[flag]
                    refer_[:,:,7:11,0,6:,:5] = refer[index][:,:,7:11,0,6:,:5]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:6,:] = refer[index][:,:,11:,0,:6,:] +similarity[flag]
                    refer_[:,:,11:,0,6:,:5] = refer[index][:,:,11:,0,6:,:5] +similarity[flag]
                    flag=flag+1   
                    refer_res.append(refer_)
                if i==10 :
                    refer_[:,:,:6,0,:,:6] = refer[index][:,:,:6,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:,:6] = refer[index][:,:,7:11,0,:,:6]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:,:6] = refer[index][:,:,11:,0,:,:6] +similarity[flag]
                    flag=flag+1
                    refer_[:,:,:6,0,:,7:11] = refer[index][:,:,:6,0,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:,7:11] = refer[index][:,:,7:11,0,:,7:11]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:,7:11] = refer[index][:,:,11:,0,:,7:11] +similarity[flag]
                    flag=flag+1
                    refer_[:,:,:6,0,:,11:] = refer[index][:,:,:6,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,7:11,0,:,11:] = refer[index][:,:,7:11,0,:,11:]+similarity[flag]
                    flag=flag+1
                    refer_[:,:,11:,0,:,11:] = refer[index][:,:,11:,0,:,11:] +similarity[flag]
                    flag=flag+1
                    refer_res.append(refer_) 
            
        atten_map = torch.cat(refer_res,dim=3)
        return atten_map

    def fusion_atten(self,a_m,trg_stru_ids,in_stru_ids):
        
        a_m_bak = a_m
        B,heads,_,_ = a_m_bak.size()
        a_m_res = []
        for index,a_m_1 in enumerate(a_m_bak):
            #a_m_1:head,HW,3HW
            a_m_1_ = torch.mean(a_m_1,dim=0).reshape(16,16,-1,16,16)
            #a_m_1_:H,W,3,H,W
            if(trg_stru_ids.shape!=in_stru_ids.shape):
                tar_stru=trg_stru_ids[index]
                in_stru = in_stru_ids[3*index:(3*index+3)] 
            else:   
                tar_stru=torch.unsqueeze(trg_stru_ids[index],0)
                in_stru = tar_stru
               
            refer_ = torch.split(a_m_1_,1,dim=2)
            a_m_1 = a_m_1.reshape(8,16,16,-1,16,16)
            refer = torch.split(a_m_1,1,dim=3)
            #先计算3HW维度上token的和
            similarity,num = self.refer_similarity(in_stru,refer_)
            #再计算内容HW上token的和得到相似度。
            similarity = self.cont_similarity(tar_stru,similarity)
            #是否使用softmax
            for i in range(len(similarity)):
                similarity[i]=similarity[i].unsqueeze(0)
            b = torch.cat(similarity)
            # b = torch.nn.functional.normalize(b,dim=0)
            # b = torch.nn.Sigmoid()(b)
            # b = torch.nn.Softmax(dim=0)(b)
            for i in range(len(similarity)):
                similarity[i]=b[i]
            atten_map = self.fusion_am(refer,in_stru,tar_stru,similarity).reshape(8,256,-1)
            a_m_res.append(atten_map) 

        a_m_fusion = torch.stack(a_m_res)

        return a_m_fusion
    
    def read_decode(self, target_style_ids, trg_sample_index, content_imgs, trg_stru_ids,in_stru_ids,reset_memory=True, \
                    reduction='mean'):
        """
        read_decode
        """
        key_matrix, value_matrix = self.read_memory(target_style_ids, trg_sample_index, reset_memory, reduction=reduction)#[B,C,H,W]
        content_feats = self.content_encoder(content_imgs) #B,C,H,W
        content_feats = content_feats
        content_feats_permute = content_feats.transpose(1,2).transpose(2,3) #B,H,W,C
        batch, h, w, channel = content_feats_permute.shape
        d_channel = int(channel / self.num_heads)
        content_feats_reshape = torch.reshape(content_feats_permute, (batch, h*w, channel)) #B, HW, C
        query_matrix = self.get_kqv_matrix(content_feats_reshape, self.linears_query)

        query_matrix = query_matrix
        query_matrix = torch.reshape(query_matrix, (batch, h*w, self.num_heads, d_channel)) #[B, HW, num_heads, C/num_heads]
        query_matrix = query_matrix.transpose(1,2).transpose(2,3) #[B, num_heads, C/num_heads, HW]
        
        ######### attention ########
        attention_mask = torch.matmul(key_matrix, query_matrix) #[B, num_heads, 3HW, HW]
        attention_mask = torch.transpose(attention_mask, 2,3) / math.sqrt(h*w)#[B, num_heads, HW, 3HW]
        attention_mask_ori = attention_mask
        attention_mask_ori = F.softmax(attention_mask_ori, dim = -1) 
        ######### SSEM ########
        attention_mask = self.fusion_atten(attention_mask,trg_stru_ids,in_stru_ids)
        attention_mask = F.softmax(attention_mask, dim = -1) 
        value_mask = torch.matmul(attention_mask, value_matrix) #[B, num_heads, HW, C/num_heads] 
        value_mask = torch.transpose(value_mask,2,3) #[B, num_heads, C/num_heads, HW]
        value_mask = torch.reshape(value_mask, (batch, channel, -1)) 
        value_mask = torch.transpose(value_mask,1,2) #[B, HW, C]    
        value_mask = self.layer_norm(value_mask)
        value_mask = torch.transpose(value_mask,1,2)
        ######### index prediction ########
        feat_scs = torch.reshape(value_mask, (batch, channel, h, w))
        fusion = feat_scs
        fusion_ = fusion.flatten(2).permute(2,0,1)
        pos = self.position_emb.unsqueeze(1).repeat(1,fusion_.shape[1],1)
        for layer in self.former:
            fusion_ = layer(fusion_,query_pos = pos)
        indice_out = self.mlp_head(fusion_).permute(1,0,2).reshape(-1,1024)
        indice_out_ = torch.argmax(indice_out,dim=-1)#shape[2048]
        z_q_x = self.vqgan.quantize.forward_with_indice(fusion,indice_out_)
        out = self.vqgan.decode(z_q_x)

        if reset_memory:
            self.reset_memory()
        return out, fusion,attention_mask,z_q_x,indice_out

    def infer(self, in_style_ids, in_imgs,in_imgs_crose,in_imgs_fine, trg_style_ids, style_sample_index, trg_sample_index, content_imgs, in_stru_ids,trg_stru_ids,
              reduction="mean"):
        """
        infer
        """
        in_style_ids = in_style_ids.cuda()
        in_imgs = in_imgs.cuda()
        trg_style_ids = trg_style_ids.cuda()
        content_imgs = content_imgs.cuda()
        in_imgs_crose = in_imgs_crose.cuda()
        in_imgs_fine = in_imgs_fine.cuda()
        self.encode_write_comb(in_style_ids, style_sample_index, in_imgs,in_imgs_crose,in_imgs_fine,in_stru_ids)
        out, feat_scs,a_m,z_q_x,indice_out = self.read_decode(trg_style_ids, trg_sample_index, content_imgs,trg_stru_ids,in_stru_ids,reduction=reduction)
        return out, feat_scs,a_m,z_q_x,indice_out

