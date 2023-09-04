# VQ-Font: Few-Shot Font Generation with Structure-Aware Enhancement and Quantization
## Abstract
Few-shot font generation is challenging, as it needs to capture the fine-grained stroke styles from a limited set of reference glyphs, and then transfer to other characters, which are expected to have similar styles. However, due to the diversity and complexity of Chinese font styles, the synthesized glyphs of existing methods usually exhibit visible artifacts, such as missing details and distorted strokes. In this paper, we propose a VQGAN-based framework (i.e., VQ-Font) to enhance glyph fidelity through token prior refinement and structure-aware enhancement. Specifically, we pre-train a VQGAN to encapsulate font token prior within a codebook. Subsequently, VQ-Font refines the synthesized glyphs with the codebook to eliminate the domain gap between synthesized and real-world strokes. Furthermore, our VQ-Font leverages the inherent design of Chinese characters, where structure components such as radicals and character components are combined in specific arrangements, to recalibrate fine-grained styles based on references. This process improves the matching and fusion of styles at the structure level. Both modules collaborate to enhance the fidelity of the generated fonts. Experiments on a collected font dataset show that our VQ-Font outperforms the competing methods both quantitatively and qualitatively, especially in generating challenging styles.  
Paper Link:[arxiv](https://arxiv.org/pdf/2308.14018.pdf)
## Dependencies
>python >= 3.7  
 torch >= 1.10.0  
 torchvision >= 0.11.0  
 sconf >= 0.2.3  
 lmdb >= 1.2.1
## Data Preparation
### 1.Images
Download '.ttf' font files from [字库](https://www.foundertype.com/), and then generate font images using the downloaded files. Arrange the fonts according to the following structural format.
>--|Font1  
>--|Font2  
>   --|char1.png  
>   --|char2.png  
>   --|...  
>--|...  
### 2.Content-Reference mapping
### 3.Struture document
### 4.Build lmdb environment
## Pre-train VQGAN
## Train VQ-FonT
## Infer VQ-Font
## Acknowledgements

