# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh   \
# co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data.py 8 \
# # --resume work_dirs/co_dino_5scale_swin_l_16xb1_1x_coco_dual_more_data/


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh   \
# projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data.py 8 \
# --resume work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data/epoch_10.pth 


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh   \
# projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data.py 8 \
# # --resume work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang/epoch_8.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh   \
# projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_c2former_more_data_offset.py 8
# python tools/train.py /root/workspace/data/dual_mmdetection/mmdetection/projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_c2former_more_data_offset.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh   \
projects/CO_DETR/configs/codino/co_dino_5scale_dinov2_b_lsj_8xb2_1x_gaiic_dual_stream_more_data_albu_with_Vis_3cls_dark_enhance.py 8
# python tools/train.py projects/CO_DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_gaiic_dual_stream_more_data_albu_with_Vis_3cls_dark_enhance_inference_vis.py