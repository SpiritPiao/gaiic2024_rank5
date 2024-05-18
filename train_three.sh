# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh   \
# co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data.py 8 \
# # --resume work_dirs/co_dino_5scale_swin_l_16xb1_1x_coco_dual_more_data/


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh   \
# projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data.py 8 \
# --resume work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data/epoch_10.pth 

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./tools/dist_train.sh   \
projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_three_stream_o365_yang_more_data_albu.py 7 \
--resume work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_three_stream_o365_yang_more_data_albu/epoch_2.pth
