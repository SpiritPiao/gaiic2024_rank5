CONFIG="/root/workspace/data/dual_mmdetection/mmdetection/projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data.py"
RESULT="analysis_results.pkl"

# python tools/test.py   \
#     $CONFIG \
#     work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data/epoch_16.pth \
#     --out $RESULT

# python tools/analysis_tools/my_confusion_matrix.py \
#     $CONFIG \
#     $RESULT \
#     ./analysis_results

python tools/analysis_tools/my_confusion_matrix.py \
    $CONFIG \
    $RESULT \
    ./analysis_results \
    --norm

# python tools/analysis_tools/eval_metric.py \
#     $CONFIG \
#     $RESULT
