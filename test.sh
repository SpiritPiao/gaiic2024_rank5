

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh   \
# projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_albu.py  \
# work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_albu/epoch_9.pth 8 --tta

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./tools/dist_test.sh   \
projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_albu.py  \
work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_albu/epoch_9.pth 7