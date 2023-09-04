from itertools import chain
import copy
from PIL import Image, ImageFile
import numpy as np
import random
import os
import json
# import paddle
import torch
from torch.utils.data import Dataset, DataLoader
from .lmdbutils import read_data_from_lmdb
import threading
import cv2
from skimage import morphology
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CombTrainDataset(Dataset):
    """
    CombTrainDataset
    """
    def __init__(self, env, env_get, avails, content_reference_json, content_font, transform=None):
        self.env = env
        self.env_get = env_get
                
        with open(content_reference_json, 'r') as f:
            self.cr_mapping = json.load(f)
        
        self.avails = avails
        #unis：content characters
        self.unis = sorted(list(self.cr_mapping.keys()))
        #font list without characters
        self.fonts = list(self.avails)
        self.n_unis = len(self.unis)
        print ('number of unis: ', self.n_unis)
        self.n_fonts = len(self.fonts)
        print ('number of fonts: ', self.n_fonts)
        self.transform = transform
        self.content_font = content_font
        
    #return font cr_mapping
    def sample_pair_style(self, font, trg_unis, avail_unis):
        trg_uni = trg_unis[0] 
        style_unis = self.cr_mapping[trg_uni]

        try:           
            imgs_ske = [np.asarray(read_data_from_lmdb(self.env,f'{font}_{uni}')['img']) for uni in style_unis]
            for i in range(len(imgs_ske)):
                _,binary = cv2.threshold(imgs_ske[i],127,255,cv2.THRESH_BINARY_INV)
                binary[binary==255] = 1
                skeleton0 = morphology.skeletonize(binary)
                imgs_ske[i] = (255-skeleton0.astype(np.uint8)*255)
                # cv2.imwrite("skeleton1.png",imgs_ske[i])
            
            imgs_ske = torch.cat([self.transform(Image.fromarray(img)) for img in imgs_ske])
            imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis]) 
        except:
            return None, None
        # imgs = paddle.concat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis])

        # print(1)
        return imgs,imgs_ske ,list(style_unis)
        
    
    #random return a character both in train font and cr_mapping.keys()
    def random_get_trg(self, avails, font_name):
        target_list = list(set.intersection(set(avails[font_name]), set(self.unis)))
        trg_uni = random.choice(target_list)
        return [trg_uni]
    
    
            
    def __getitem__(self, index):
        #randomly choose a font
        font_idx = index % self.n_fonts
        font_name = self.fonts[font_idx]
        while True:
            #randomly choose target
            trg_unis = self.random_get_trg(self.avails, font_name)
            sample_index = torch.tensor([index])
            
            avail_unis = self.avails[font_name]
            style_imgs, style_imgs_ske,style_unis = self.sample_pair_style(font_name, trg_unis, avail_unis)
             
            if style_imgs is None:
                continue
                
            #add trg_imgs
            trg_imgs = torch.cat([self.env_get(self.env, font_name, uni, self.transform)
                                  for uni in trg_unis])
            
            trg_uni_ids = [self.unis.index(uni) for uni in trg_unis]
            font_idx = torch.tensor([font_idx])
            
            content_imgs = torch.cat([self.env_get(self.env, self.content_font, uni, self.transform)
                                      for uni in trg_unis]).unsqueeze_(1)
           
            content_imgs_ske = [np.asarray(read_data_from_lmdb(self.env,f'{self.content_font}_{uni}')['img']) for uni in trg_unis]
            for i in range(len(content_imgs_ske)):
                _,binary = cv2.threshold(content_imgs_ske[i],127,255,cv2.THRESH_BINARY_INV)
                binary[binary==255] = 1
                skeleton0 = morphology.skeletonize(binary)
                content_imgs_ske[i] = (255-skeleton0.astype(np.uint8)*255)
                # cv2.imwrite("content_imgs_ske.png",content_imgs_ske[i])
            
            content_imgs_ske = torch.cat([self.transform(Image.fromarray(img)) for img in content_imgs_ske])
           
            # print('sss',len(style_imgs)) 3
            # print('f:',font_idx)
            # print('s:',sample_index)
            # print(trg_imgs.shape,content_imgs.shape)[1, 128, 128][1, 1, 128, 128]

            ret = (
                torch.repeat_interleave(font_idx, len(style_imgs)),
                style_imgs,
                style_imgs_ske,
                torch.repeat_interleave(font_idx, len(trg_imgs)),
                torch.tensor(trg_uni_ids),
                trg_imgs,
                content_imgs,
                content_imgs_ske,
                trg_unis,
                torch.repeat_interleave(sample_index, len(style_imgs)), #style sample index
                sample_index #trg sample index
            )
            
            return ret

    def __len__(self):
        return sum([len(v) for v in self.avails.values()])

    @staticmethod
    def collate_fn(batch):
        (style_ids, style_imgs,style_imgs_ske,
         trg_ids, trg_uni_ids, trg_imgs, content_imgs,content_imgs_ske, trg_unis, style_sample_index, trg_sample_index) = zip(*batch) 
        
        #print (style_comp_ids)

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(style_imgs_ske).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_uni_ids),
            torch.cat(trg_imgs).unsqueeze_(1),
            torch.cat(content_imgs),
            torch.cat(content_imgs_ske).unsqueeze_(1),
            trg_unis,
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index)
        )
        
        return ret

class   CombTestDataset(Dataset):
    """
    CombTestDataset
    """
    def __init__(self, env, env_get, target_fu, avails, content_reference_json, content_font, language="chn",
                 transform=None, ret_targets=True):

        self.fonts = list(target_fu)
        self.n_uni_per_font = len(target_fu[list(target_fu)[0]])
        self.fus = [(fname, uni) for fname, unis in target_fu.items() for uni in unis]
        self.unis = sorted(set.union(*map(set, avails.values())))
        self.env = env
        self.env_get = env_get
        self.avails = avails
            
        with open(content_reference_json, 'r') as f:
            self.cr_mapping = json.load(f)
            
        self.train_unis = sorted(set.union(*map(set, self.cr_mapping.values())))
        self.transform = transform
        self.ret_targets = ret_targets
        self.content_font = content_font

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                      }

        self.to_int = to_int_dict[language.lower()]
        
    def sample_pair_style(self, trg_uni, avail_unis):
            
        if trg_uni not in self.cr_mapping:
            style_unis = random.sample(avail_unis, 3)
        else:
            style_ref = self.cr_mapping[trg_uni]
            style_unis = []
            for ref in style_ref:
                if ref in avail_unis:
                    style_unis.append(ref)
                else:
                    style_unis.append(random.choice(avail_unis))
        return list(style_unis)
    

    def __getitem__(self, index):
        font_name, trg_uni = self.fus[index]
        font_idx = self.fonts.index(font_name)
        sample_index = torch.tensor([index])
        
        avail_unis = self.avails[font_name]
        style_unis = self.sample_pair_style(trg_uni, avail_unis)
        
        try:
            
            imgs_ske = [np.asarray(read_data_from_lmdb(self.env,f'{font_name}_{uni}')['img']) for uni in style_unis]
            for i in range(len(imgs_ske)):
                _,binary = cv2.threshold(imgs_ske[i],127,255,cv2.THRESH_BINARY_INV)
                binary[binary==255] = 1
                skeleton0 = morphology.skeletonize(binary)
                imgs_ske[i] = (255-skeleton0.astype(np.uint8)*255)
                # cv2.imwrite("skeleton1.png",imgs_ske[i])
            b = [self.transform(Image.fromarray(img)) for img in imgs_ske]
            
            
            # print(style_imgs_ske.shape,'style_imgs_ske')
            # aaa
            a = [self.env_get(self.env, font_name, uni, self.transform) for uni in style_unis]
            
        except:
            print (font_name, style_unis)

        style_imgs = torch.stack(a)
        style_imgs_ske =  torch.stack(b)            
        # print(a[0].shape,b[0].shape,style_imgs.shape,style_imgs_ske.shape)
        
        font_idx = torch.tensor([font_idx])
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])
        
        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)
        
        content_imgs_ske = np.asarray(read_data_from_lmdb(self.env,f'{self.content_font}_{trg_uni}')['img'])
        _,binary = cv2.threshold(content_imgs_ske,127,255,cv2.THRESH_BINARY_INV)
        binary[binary==255] = 1
        skeleton0 = morphology.skeletonize(binary)
        content_imgs_ske = (255-skeleton0.astype(np.uint8)*255)
        content_imgs_ske = self.transform(Image.fromarray(content_imgs_ske))

        
        ret = (
            torch.repeat_interleave(font_idx, len(style_imgs)),
            style_imgs,
            style_imgs_ske,
            font_idx,
            trg_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)), #style sample index
            sample_index, #trg sample index
            content_img,
            content_imgs_ske,
            
        )
        
        if self.ret_targets:
            try:
                trg_img = self.env_get(self.env, font_name, trg_uni, self.transform)
            except:
                trg_img = torch.ones(1, 128, 128)
            ret += (trg_img, )

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_imgs,style_imgs_ske, trg_ids, trg_unis, style_sample_index, trg_sample_index, content_imgs,content_imgs_ske, *left = list(zip(*batch))
        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs),
            torch.cat(style_imgs_ske),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1),
            torch.cat(content_imgs_ske).unsqueeze_(1),
            
        )

        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret
    
    
class FixedRefDataset(Dataset):
    '''
    FixedRefDataset
    '''
    def __init__(self, env, env_get, target_dict, ref_unis, k_shot, content_reference_json, content_font, language="chn",  transform=None, ret_targets=True):
        '''
        ref_unis: target unis
        target_dict: {style_font: [uni1, uni2, uni3]} gen_fonts:gen_unis
        '''
        self.target_dict = target_dict
        self.ref_unis = sorted(ref_unis)
        self.fus = [(fname, uni) for fname, unis in target_dict.items() for uni in unis]
        self.k_shot = k_shot
        with open(content_reference_json, 'r') as f:
            self.cr_mapping = json.load(f)
            
        self.content_font = content_font
        self.fonts = list(target_dict)

        self.env = env
        self.env_get = env_get
        
        self.transform = transform
        self.ret_targets = ret_targets

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                      }

        self.to_int = to_int_dict[language.lower()]
        
    def sample_pair_style(self, font, trg_uni):
        assert trg_uni in self.cr_mapping, "infer uni is not in your content reference map"
        style_unis = self.cr_mapping[trg_uni]
        imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis])
        return imgs, list(style_unis)


    def __getitem__(self, index):
        fname, trg_uni = self.fus[index]
        sample_index = torch.tensor([index])
        
        fidx = self.fonts.index(fname)
        avail_unis = list(set(self.ref_unis) - set([trg_uni]))
        style_imgs, style_unis = self.sample_pair_style(fname, trg_uni)
        
        fidces = torch.tensor([fidx])
        trg_dec_uni =torch.tensor([self.to_int(trg_uni)])
        style_dec_uni = torch.tensor([self.to_int(style_uni) for style_uni in style_unis])
        
        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)
        ret = (
            torch.repeat_interleave(fidces, len(style_imgs)), #fidces,
            style_imgs,
            fidces,
            trg_dec_uni,
            style_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img
        )

        if self.ret_targets:
            trg_img = self.env_user_get(self.env_user, fname, trg_uni, self.transform)
            ret += (trg_img, )

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_imgs, trg_ids, trg_unis, style_unis, style_sample_index, trg_sample_index, content_imgs, *left = \
            list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_unis),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1)
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret


class FixedRefDataset_random(Dataset):
    '''
    FixedRefDataset
    '''

    def __init__(self, env, env_get, target_dict, ref_unis, k_shot, content_reference_json, content_font,
                 language="chn", transform=None, ret_targets=True):
        '''
        ref_unis: target unis
        target_dict: {style_font: [uni1, uni2, uni3]} gen_fonts:gen_unis
        '''
        self.target_dict = target_dict
        self.ref_unis = sorted(ref_unis)
        self.fus = [(fname, uni) for fname, unis in target_dict.items() for uni in unis]
        self.k_shot = k_shot
        with open(content_reference_json, 'r') as f:
            self.cr_mapping = json.load(f)

        self.content_font = content_font
        self.fonts = list(target_dict)

        self.env = env
        self.env_get = env_get

        self.transform = transform
        self.ret_targets = ret_targets

        to_int_dict = {"chn": lambda x: int(x, 16),
                       "kor": lambda x: ord(x),
                       "thai": lambda x: int("".join([f'{ord(each):04X}' for each in x]), 16)
                       }

        self.to_int = to_int_dict[language.lower()]

    def sample_pair_style(self, font, trg_uni):
        assert trg_uni in self.cr_mapping, "infer uni is not in your content reference map"
        style_unis = self.ref_unis
        print('self.ref_unis:',self.ref_unis)
        imgs = torch.cat([self.env_get(self.env, font, uni, self.transform) for uni in style_unis])
        return imgs, list(style_unis)

    def __getitem__(self, index):
        fname, trg_uni = self.fus[index]
        sample_index = torch.tensor([index])

        fidx = self.fonts.index(fname)
        avail_unis = list(set(self.ref_unis) - set([trg_uni]))
        style_imgs, style_unis = self.sample_pair_style(fname, trg_uni)

        fidces = torch.tensor([fidx])
        trg_dec_uni = torch.tensor([self.to_int(trg_uni)])
        style_dec_uni = torch.tensor([self.to_int(style_uni) for style_uni in style_unis])

        content_img = self.env_get(self.env, self.content_font, trg_uni, self.transform)
        ret = (
            torch.repeat_interleave(fidces, len(style_imgs)),  # fidces,
            style_imgs,
            fidces,
            trg_dec_uni,
            style_dec_uni,
            torch.repeat_interleave(sample_index, len(style_imgs)),
            sample_index,
            content_img
        )

        if self.ret_targets:
            trg_img = self.env_user_get(self.env_user, fname, trg_uni, self.transform)
            ret += (trg_img,)

        return ret

    def __len__(self):
        return len(self.fus)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_imgs, trg_ids, trg_unis, style_unis, style_sample_index, trg_sample_index, content_imgs, *left = \
            list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_unis),
            torch.cat(style_unis),
            torch.cat(style_sample_index),
            torch.cat(trg_sample_index),
            torch.cat(content_imgs).unsqueeze_(1)
        )
        if left:
            trg_imgs = left[0]
            ret += (torch.cat(trg_imgs).unsqueeze_(1),)

        return ret

