export CUDA_VISIBLE_DEVICES=7
python3 inference.py ./cfgs/custom_infer.yaml \
--weight /data/yms/formerfont_vqgan/results/baseline/checkpoints/codebook_512/340000-codebook_512.pdparams  \
--content_font /data/yms/datasets/font_png_select/content/FZShengSKSJW_Xian   \
--img_path /data/yms/datasets/font_png_select/valid_sfuf \
--saving_root 512_bak 
