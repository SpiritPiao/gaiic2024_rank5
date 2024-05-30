

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./tools/dist_test.sh   \
# projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_albu_with_Vis_3cls.py  \
# work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_albu_with_Vis_3cls/epoch_6.pth 7 --tta

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./tools/dist_test.sh   \
# projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_albu_with_Vis_3cls_drak_enhance.py  \
# work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_albu_with_Vis_3cls_drak_enhance/0519_dark_529.pth 7

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh   \
/root/workspace/data/dual_mmdetection/mmdetection/projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_albu_with_Vis_3cls_drak_enhance_cbpkiv3_rotate.py  \
/root/workspace/data/dual_mmdetection/mmdetection/work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_albu_with_Vis_3cls_drak_enhance_cbpkiv3_rotate/epoch_7.pth 8 --tta