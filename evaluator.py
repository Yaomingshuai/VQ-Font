import tqdm
import cv2
import torch
import numpy as np
import os
import utils
from torchvision.utils import make_grid, save_image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import torch.nn.functional as F
import lpips
import json
with open('/data/yms/formerfont_vqgan/meta/stru.json','r') as f:
    stru_map = json.load(f,strict=False)

with open('/data/yms/formerfont_vqgan/meta/cr_mapping.json','r') as f:
    cr_map = json.load(f,strict=False)
with open('/data/yms/formerfont_vqgan/meta/de.json','r') as f:
    de = json.load(f,strict=False)
def paddle_eval(val_fn):
    @torch.no_grad()
    def decorated(self, gen, *args, **kwargs):
        gen.eval()
        ret = val_fn(self, gen, *args, **kwargs)
        gen.train()
        return ret

    return decorated

def batch_psnr(img, imclean, data_range):
    """
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += peak_signal_noise_ratio(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
                    data_range=2)
    return psnr/img_cpu.shape[0]

def batch_ssim(img, imclean):

    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += structural_similarity(imgclean[i, :, :, :].transpose(1,2,0), img_cpu[i, :, :, :].transpose(1,2,0), \
                    data_range=2,channel_axis=2)
    return psnr/img_cpu.shape[0]

class Evaluator:
    """
    Evaluator
    """

    def __init__(
        self,
        env,
        env_get,
        cfg,
        logger,
        writer,
        batch_size,
        transform,
        content_font,
        use_half=False,
    ):

        self.env = env
        self.env_get = env_get
        self.logger = logger
        self.batch_size = batch_size
        self.transform = transform
        self.writer = writer
        self.k_shot = cfg.kshot
        self.content_font = content_font
        self.use_half = use_half
        self.size = cfg.input_size
        self.loss_fn_alex = lpips.LPIPS(net='alex') 
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        # self.perceptual_loss = LPIPS().eval()
    
    def cp_validation(self, gen, cv_loaders, step, reduction="mean", ext_tag=""):
        """
        cp_validation
        """

        for tag, loader in cv_loaders.items():
            self.comparable_val_saveimg(
                gen,
                loader,
                step,
                kshot=self.k_shot,
                tag=f"comparable_{tag}_{ext_tag}",
                reduction=reduction,
            )

    @paddle_eval
    def comparable_val_saveimg(
        self, gen, loader, step, kshot=3, tag="comparable", reduction="mean"
    ):
        n_row = loader.dataset.n_uni_per_font * kshot
        compare_batches ,metrics= self.infer_loader(gen, loader, kshot, reduction=reduction)

        comparable_grid = utils.make_comparable_grid(*compare_batches[::-1], nrow=n_row)
        self.writer.add_image(metrics,tag, comparable_grid, step)
        return comparable_grid     

    @paddle_eval
    def infer_loader(self, gen, loader, kshot, reduction="mean"):
        outs = []
        trgs = []
        styles = []

        for (
            i,
            (
                in_style_ids,
                in_imgs,
                in_imgs_ske,
                trg_style_ids,
                trg_unis,
                style_sample_index,
                trg_sample_index,
                content_imgs,
                content_imgs_ske,
                *trg_imgs,
            ),
        ) in enumerate(loader):
            if self.use_half:
                in_imgs = in_imgs.half()
                content_imgs = content_imgs.half()
            in_imgs_fine = torch.nn.functional.interpolate(in_imgs,scale_factor=0.8,mode='bilinear')
            in_imgs_crose = torch.nn.functional.interpolate(in_imgs,scale_factor=1.2,mode='bilinear')
            
            trg_unis_bak=trg_unis
            trg_unis = []
            for i in trg_unis_bak:
                trg_unis.append([str(hex(int(i))).upper()[2:]])             
            in_styles_unis = []
            for i in trg_unis:
                in_styles_unis.append(cr_map[i[0]])

            #获取结构信息
            trg_stru_ids = []
            for i in trg_unis:
                trg_stru_ids.append(stru_map[i[0]])
            trg_stru_ids = torch.tensor(trg_stru_ids).cuda()
            in_stru_ids=[]
            for k in in_styles_unis:
                for i in range(3):
                    in_stru_ids.append(stru_map[k[i]])
            in_stru_ids = torch.tensor(in_stru_ids).cuda()
            
            #获取部件信息
            trg_comp_ids = []
            for i in trg_unis:
                trg_comp_ids.append(de[i[0]])
            in_comp_ids=[]
            for k in in_styles_unis:
                for i in range(3):
                    in_comp_ids.append(de[k[i]])

            out, _, _,_ ,_= gen.infer(
                in_style_ids,
                in_imgs,
                in_imgs_crose,
                in_imgs_fine,
                trg_style_ids,
                style_sample_index,
                trg_sample_index,
                content_imgs,
                in_stru_ids,
                trg_stru_ids,
                reduction=reduction,
            )

            batch_size = out.shape[0]
            out_images = out.detach().cpu().numpy()
            out_duplicate = np.ones((batch_size * kshot, 1, self.size, self.size))
            for idx in range(batch_size):
                for j in range(kshot):
                    out_duplicate[idx * kshot + j, ...] = out_images[idx, ...]
            outs.append(torch.Tensor(out_duplicate))

            for style_img in in_imgs:
                style_duplicate = np.ones((1, 1, self.size, self.size))
                style_duplicate[:, :, :, :] = style_img.unsqueeze(1).detach().cpu()
                styles.append(torch.Tensor(style_duplicate))

            if trg_imgs:
                trg_images = trg_imgs[0].detach().cpu().numpy()
                trg_duplicate = np.zeros((batch_size * kshot, 1, self.size, self.size))
                for idx in range(batch_size):
                    for j in range(kshot):
                        trg_duplicate[idx * kshot + j, ...] = trg_images[idx, ...]
                trgs.append(torch.Tensor(trg_duplicate))

        ret = (torch.cat(outs).float(),)
        if trgs:
            ret += (torch.cat(trgs).float(),)

        ret += (torch.cat(styles).float(),)
        
        psnr = batch_psnr(ret[1],ret[0],data_range=1)
        l1 = F.l1_loss(ret[1].detach().cpu(), ret[0].detach().cpu(), reduction="mean").item()
        Rmse = torch.sqrt(F.mse_loss(ret[1].detach().cpu(), ret[0].detach().cpu(), reduction="mean")).item()
        ssim = batch_ssim(ret[1],ret[0])
        lpips_alex = self.loss_fn_alex(ret[1].detach().cpu(),ret[0].detach().cpu()).mean().item()
        lpips_vgg = self.loss_fn_vgg(ret[1].detach().cpu(),ret[0].detach().cpu()).mean().item()
        print('l1:','%.3f'%l1,'Rmse:','%.3f'%Rmse,'psnr:','%.3f'%psnr,'ssim:','%.3f'%ssim,'lpips_alex:','%.3f'%lpips_alex,'lpips_vgg:','%.3f'%lpips_vgg)
        a = {}
        a['Rmse']=Rmse
        a['psnr']=psnr
        a['ssim']=ssim
        a['l1']=l1
        a['lpips_vgg']=lpips_vgg
        a['lpips_alex']=lpips_alex
        
        return ret,a

    def normalize(self, tensor, eps=1e-5):
        """ Normalize tensor to [0, 1] """
        # eps=1e-5 is same as make_grid in torchvision.
        minv, maxv = tensor.min(), tensor.max()
        tensor = (tensor - minv) / (maxv - minv + eps)
        return tensor

    @paddle_eval
    def save_each_imgs(self, gen, loader, ori_img_root, save_dir, reduction="mean"):
        """
        save_each_imgs
        """
        font_name = os.path.basename(save_dir)
        output_folder = os.path.join(save_dir, "images")
        os.makedirs(output_folder, exist_ok=True)
        ch_list_check = []
        for (in_style_ids,in_imgs,trg_style_ids,trg_unis,style_unis,style_sample_index,trg_sample_index,content_imgs,) in tqdm.tqdm(loader):
            if self.use_half:
                in_imgs = in_imgs.half()
                content_imgs = content_imgs.half()
          
            trg_unis_bak=trg_unis
            trg_unis = []
            for i in trg_unis_bak:
                trg_unis.append([str(hex(int(i))).upper()[2:]])             

            in_styles_unis = []
            for i in trg_unis:
                in_styles_unis.append(cr_map[i[0]])

            #获取结构信息
            trg_stru_ids = []
            for i in trg_unis:
                trg_stru_ids.append(stru_map[i[0]])
            trg_stru_ids = torch.tensor(trg_stru_ids).cuda()
            in_stru_ids=[]
            for k in in_styles_unis:
                for i in range(3):
                    in_stru_ids.append(stru_map[k[i]])
            in_stru_ids = torch.tensor(in_stru_ids).cuda()
            
            #获取部件信息
            trg_comp_ids = []
            for i in trg_unis:
                trg_comp_ids.append(de[i[0]])
            in_comp_ids=[]
            for k in in_styles_unis:
                for i in range(3):
                    in_comp_ids.append(de[k[i]])
                    
            out, _, A_M,_ ,_= gen.infer(in_style_ids,in_imgs,in_imgs,in_imgs,trg_style_ids,style_sample_index,trg_sample_index,content_imgs,in_stru_ids,trg_stru_ids,reduction=reduction,)
            area1={'仇：人':[69,84,85,100,115,116,131,132,147,148,164,180,196,212]}
            area = {k:v for d in [area1] for k, v in d.items()}
            for idex in range(len(list(area.keys()))):
                a = []
                for i in torch.split(A_M, 1, dim=1):
                    i = i[:,:,list(area.values())[idex],:,]
                    i = torch.sum(i, dim=2).reshape(-1, 3, 16,16)
                    a.append(i)
                dec_unis = trg_unis_bak.detach().cpu().numpy()
                style_dec_unis = style_unis.detach().cpu().numpy()
                font_ids = trg_style_ids.detach().cpu().numpy()
                images = out.detach().cpu()  # [B, 1, 128, 128]
                in_imgs_3 = torch.split(in_imgs, 3, dim=0)

                a_ms = a[6].detach().cpu().unsqueeze(1)
                for (
                    idx,
                    (content_img, in_style_id, dec_uni, font_id, image, a_m, in_img),
                ) in enumerate(
                    zip(content_imgs, in_style_ids, dec_unis, font_ids, images, a_ms, in_imgs_3)
                ):

                    font_name = loader.dataset.fonts[font_id]  # name.ttf
                    uni = hex(dec_uni)[2:].upper().zfill(4)
                    ch = "\\u{:s}".format(uni).encode().decode("unicode_escape")
                    gener = self.normalize(image)
                    final_img = torch.transpose(torch.clamp(gener*255, min=0, max=255), 0,1)
                    final_img = torch.transpose(final_img, 1,2).cpu().numpy()
                    if final_img.shape[-1] == 1:
                        final_img = final_img.squeeze(-1) #[128, 128]
                    dst_path = os.path.join(output_folder, ch + '.png')
                    ch_list_check.append(ch)
                    cv2.imwrite(dst_path, final_img)

                    # 生成：attention_map 
                    image,a_m = self.normalize(image),self.normalize(a_m)
                    image, content_img = image.squeeze(0), content_img.squeeze(0)
                    in_img = self.normalize(in_img.permute(1,0,2,3))
                    a_m, in_img = torch.split(a_m.squeeze(0),1,dim=0),torch.split(in_img.squeeze(0),1,dim=0)
                    a_m, in_img = torch.cat([a_m[i].squeeze(0) for i in range(3)],dim=1),torch.cat([in_img[i].squeeze(0) for i in range(3)],dim=1) 
                    a_m, in_img = torch.clamp(a_m * 255, min=0, max=255),torch.clamp(in_img * 255, min=0, max=255)
                    image, content_img = torch.clamp(image * 255, min=0, max=255),torch.clamp(content_img * 255, min=0, max=255)
                    a_m,  in_img, image, content_img = a_m.cpu().numpy(),in_img.cpu().numpy(),image.cpu().numpy(),content_img.cpu().numpy()

                    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                    content_img = cv2.cvtColor(content_img,cv2.COLOR_GRAY2RGB)
                    a_m = cv2.applyColorMap(a_m.astype('uint8'),2)
                    in_img = cv2.cvtColor(in_img, cv2.COLOR_GRAY2RGB)
                    a_m = cv2.resize(a_m,(128*3,128))
                    in_img = cv2.resize(in_img.astype('uint8'),(128*3,128))
                    img_add = cv2.addWeighted(in_img, 0.3, a_m, 0.7, 0)#0.8 0.6////0.6 0.8
                    
                    import numpy as np
                    gt_img = cv2.imread(os.path.join('/data/yms/datasets/font_png_select/valid_sfuf',output_folder.split('/')[1],ch+'.png'),flags=1)
                    final_img = np.hstack([content_img,image,img_add,gt_img])
                    gt_path = output_folder.replace('images','gt_images')
                    os.makedirs(os.path.join(gt_path), exist_ok=True)
                    gt_path = os.path.join(gt_path, ch+'.png')
                    cv2.imwrite(gt_path, gt_img)
                    b = output_folder.replace('images','attention_map')
                    os.makedirs(os.path.join(b), exist_ok=True)
                    b = os.path.join(b, ch+'--'+list(area.keys())[idex] +'.png')
                    if ch in ['仇'] and ch ==list(area.keys())[idex][:1]:
                        ch_list_check.append(ch)
                        cv2.imwrite(b, final_img)
            print("num_saved_img: ", len(ch_list_check))
        return output_folder

