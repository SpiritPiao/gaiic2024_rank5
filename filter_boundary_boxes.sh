RESULT_JSON=/root/workspace/data/dual_mmdetection/mmdetection/dual_test_result.bbox.json
ANNOTATION=/root/workspace/data/GAIIC2024/instances_test2017.json

OUTPUT_DIR=/root/workspace/data/dual_mmdetection/mmdetection

python tools/analysis_tools/filter_board_boxes.py \
      ${RESULT_JSON} \
       --annotation ${ANNOTATION} \
       --save-filter-results \
       --fiter-cids 1 \
       --fiter-hw-ratio 12 \
       --fiter-area 300 \
       --boundary-dis 1 \
       --out-dir ${OUTPUT_DIR}
