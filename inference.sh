# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh   \
# co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data.py 8 \
# # --resume work_dirs/co_dino_5scale_swin_l_16xb1_1x_coco_dual_more_data/

python demo/image_demo.py root/workspace/data/GAIIC2024/test/trg projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data.py  \
# --weights work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data/epoch_16.pth \
