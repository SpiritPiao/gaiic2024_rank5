CONFIG="/root/workspace/data/dual_mmdetection/mmdetection/projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_albu_train.py"
RESULT="pred_result.pkl"
RESULT="/root/workspace/data/dual_mmdetection/mmdetection/0516_4947.pkl"

SAVE_DIR="./co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_albu_train"
mkdir ./analysis_results/${SAVE_DIR}

# python tools/test.py   \
#     $CONFIG \
#     work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/0510_4907.pth \
#     --out $RESULT

python tools/analysis_tools/my_confusion_matrix.py \
    $CONFIG \
    $RESULT \
    ./analysis_results/${SAVE_DIR}

python tools/analysis_tools/my_confusion_matrix.py \
    $CONFIG \
    $RESULT \
    ./analysis_results/${SAVE_DIR} \
    --norm

python tools/analysis_tools/eval_metric.py \
    $CONFIG \
    $RESULT
