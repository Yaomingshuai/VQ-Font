# VQ-Font: Few-Shot Font Generation with Structure-Aware Enhancement and Quantization
## Abstract
Few-shot font generation is challenging, as it needs to capture the fine-grained stroke styles from a limited set of reference glyphs, and then transfer to other characters, which are expected to have similar styles. However, due to the diversity and complexity of Chinese font styles, the synthesized glyphs of existing methods usually exhibit visible artifacts, such as missing details and distorted strokes. In this paper, we propose a VQGAN-based framework (i.e., VQ-Font) to enhance glyph fidelity through token prior refinement and structure-aware enhancement. Specifically, we pre-train a VQGAN to encapsulate font token prior within a codebook. Subsequently, VQ-Font refines the synthesized glyphs with the codebook to eliminate the domain gap between synthesized and real-world strokes. Furthermore, our VQ-Font leverages the inherent design of Chinese characters, where structure components such as radicals and character components are combined in specific arrangements, to recalibrate fine-grained styles based on references. This process improves the matching and fusion of styles at the structure level. Both modules collaborate to enhance the fidelity of the generated fonts. Experiments on a collected font dataset show that our VQ-Font outperforms the competing methods both quantitatively and qualitatively, especially in generating challenging styles.  
Paper Link: [arxiv](https://arxiv.org/pdf/2308.14018.pdf)
## Dependencies
>python >= 3.7  
 torch >= 1.10.0  
 torchvision >= 0.11.0  
 sconf >= 0.2.3  
 lmdb >= 1.2.1
## Data Preparation
### 1.Images and Characters 
Download '.ttf' font files from [字库](https://www.foundertype.com/), and then generate font images using the '.ttf' files. Select one font as the content font, and then split the remaining fonts into training set and test set, arranging the fonts according to the following structure:
>Train Font Directory  
|&emsp;--|Font_1  
|&emsp;--|Font_2  
|&emsp;&emsp;&emsp;--|char_1.png  
|&emsp;&emsp;&emsp;--|char_2.png  
|&emsp;&emsp;&emsp;--|...  
|&emsp;&emsp;&emsp;--|char_n.png  
|&emsp;--|...  
|&emsp;--|Font_n  
Test Font Directory  
Content Font Directory

At the same time, split the Chinese characters into train characters and valid characters. Then convert them into Unicode form through hex(ord(ch))[2:].upper( ) and save them to JSON files.  
>train_unis: ["5211","597D","80DC"]  
 val_unis: ["8FD1","4FA0"]
### 2.Content-Reference Mapping
Referring to the method mentioned in [Fs-Font](https://github.com/tlc121/FsFont), we first select around 100 reference characters from all Chinese characters as our reference set, and then select three reference characters for each character from the reference set. The format of C-R mapping is as shown below:  
>{content_1: [ref_1, ref_2, ref_3, ...], content2: [ref_1, ref_2, ref_3, ...], ...}

example:
>{"5211": ["5F62","520A","5DE7"],"597D": ["5B59","5987","59E5"],"80DC": ["80A0","7272","81C0"],"8FD1": ["65A5","65B0","8FC5"],"4FA0": ["62F3","4EC6","4FED"]}

### 3.Struture Tags
Chinese characters can be divided into approximately 12 structure types, which we can represent with numbers 0 to 11. Then we add structure tags for each Chinese character：
>{Uni_1: stru_tag_1, Uni_2: stru_tag_2, ...}

example:  
>{ "81C0": 0, "8FD1": 8, "65A5": 3, "65B0": 4, "8FC5": 8, "4FA0": 4, "62F3": 0, "4EC6": 4, "4FED": 4, ...}

train_unis.json、val_unis.json、cr_mapping.json and stru.json are all in ./meta.
### 4.Build Lmdb Environment
Run Scripts
```
python3 ./build_dataset/build_meta4train.py 
--saving_dir ./results/your_task_name/ 
--content_font path\to\content 
--train_font_dir path\to\training_font 
--val_font_dir path\to\validation_font 
--seen_unis_file path\to\train_unis.json 
--unseen_unis_file path\to\val_unis.json
```
## Pre-train VQGAN
First, store the paths of the training images and test images in vqgan_data/train.txt and vqgan_data/valid.txt respectively.  
Run Scripts   
```
python taming/main.py --base vqgan/custom_vqgan.yaml -t True
```
Keys  
* base: path to config file for training VQGAN  
* t: switching to the training pattern mode  
After obtaining the pre-trained model, we put it into ./vqgan.  
## Train VQ-Font
Run Scripts
```
python3 train.py 
    task_name
    cfgs/custom.yaml
    --resume \path\to\your\pretrain_model.pdparams
```
## Infer VQ-Font
Run Scripts  
```
python3 inference.py ./cfgs/custom.yaml 
--weight path\to\saved_weight.pdparams
--content_font path\to\content 
--img_path path\to\reference 
--saving_root path\to\saving_folder
```
## Acknowledgements
Our code is modified based on the [FS-Font](https://github.com/tlc121/FsFont) and [LF-Font](https://github.com/clovaai/lffont).

