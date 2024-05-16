import os, glob, numpy as np
import argparse

from mmengine.fileio import dump, load
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



def parse_args():
    parser = argparse.ArgumentParser(description='Fusion image \
        prediction results using Weighted \
        Boxes Fusion from multiple models.')
    parser.add_argument(
        'pred_result',
        type=str,
        # nargs='+',
        help='files of prediction results \
                    from multiple models, json format.')
    parser.add_argument('--annotation', type=str, help='annotation file path')
    parser.add_argument(
        '--fiter-cids',
        nargs="+",
        default=[1, 4],
        help='how to calculate confidence in weighted boxes in wbf.')
    parser.add_argument(
        '--fiter-hw-ratio',
        # nargs="+",
        type=float,
        default=4.0,
        help='how to calculate confidence in weighted boxes in wbf.')
    parser.add_argument(
        '--fiter-area',
        type=float,
        default=45*45 / 2,
        help='how to calculate confidence in weighted boxes in wbf.')
    parser.add_argument(
        '--boundary-dis',
        type=int,
        default=2,
        help='distance to the image boundary.')
    # parser.add_argument(
    #     '--conf-type',
    #     type=str,
    #     default='avg',
    #     help='how to calculate confidence in weighted boxes in wbf.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--save-filter-results',
        action='store_true',
        help='whether save fusion result')

    args = parser.parse_args()

    return args


def iou(box: np.ndarray, boxes: np.ndarray):
    """ 计算一个边界框和多个边界框的交并比

    Parameters
    ----------
    box: `~np.ndarray` of shape `(4, )`
        边界框

    boxes: `~np.ndarray` of shape `(n, 4)`
        其他边界框

    Returns
    -------
    iou: `~np.ndarray` of shape `(n, )`
        交并比
    """
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]

    area_boxes = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0]) * (box[3]-box[1])

    return inter / area_boxes

from copy import deepcopy
def filter_boundary_boxes(boxes, scores, labels, h, w, cids, dis_threshold, area_threshold, hw_ratio_threshold):
    cids = list(map(int, cids))
    lt_img = [dis_threshold, dis_threshold]
    rb_img = [w - dis_threshold, h - dis_threshold]
    
    boxes = np.array(boxes)
    old_boxes = deepcopy(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    
    whole_img_box = np.array(lt_img + rb_img)
    # print(whole_img_box, boxes[:10])
    ious = iou(whole_img_box, boxes)
    # print(ious[ious < 0.5])
    bound_boxes_indices = (ious < 1 - 1e-5)
    # print(np.count_nonzero(bound_boxes_indices))
    
    # print(boxes[bound_boxes_indices][:20], w, h)
    
    if len(cids) == 0:
        return old_boxes, scores, labels, 0
    
    # print(cids, np.unique(labels))
    valid_cid_indices = (labels == cids[0])
    for i in cids[1:]:
        valid_cid_indices |= (labels == i)
        
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    area = w * h
    hw_ratio = h / (w + 1e-3)
    valid_area_indices = (area < area_threshold)

    valid_hw_ratio_indices = (hw_ratio < 1 / hw_ratio_threshold) | (hw_ratio > hw_ratio_threshold)
    
    target = bound_boxes_indices & valid_cid_indices
    valid_indices = (target & valid_area_indices) | (target & valid_hw_ratio_indices)
    # try:
    #     print(scores[valid_indices].max())
    # except:
    #     pass
    boxes = old_boxes[~valid_indices]
    scores = scores[~valid_indices]
    labels = labels[~valid_indices]    

    return boxes, scores, labels, np.count_nonzero(valid_indices)


def main():
    args = parse_args()

    path = args.pred_result
    pred = load(path)
    anno = args.annotation
    predicts_raw = pred

    cocoGT = COCO(annotation_file=anno)
    predict = {
        str(image_id): {
            'bboxes_list': [],
            'scores_list': [],
            'labels_list': []
        }
        for image_id in cocoGT.getImgIds()
    }

    for pred in predicts_raw:
        p = predict[str(pred['image_id'])]
        # print(str(pred['image_id']))
        p['bboxes_list'].append(pred['bbox'])
        p['scores_list'].append(pred['score'])
        p['labels_list'].append(pred['category_id'])
        
    
    result = []
    f_cnt = 0
    cnt = 0
    for image_id, res in predict.items():
        # print(image_id)
        # print(cocoGT.imgs)
        t = cocoGT.loadImgs(int(image_id))[0]
        
        h, w = t['height'], t['width']
        bboxes, scores, labels, filte_success = filter_boundary_boxes(
            res['bboxes_list'],
            res['scores_list'],
            res['labels_list'],
            h, w,
            cids=args.fiter_cids,
            dis_threshold=args.boundary_dis,
            area_threshold=args.fiter_area,
            hw_ratio_threshold=args.fiter_hw_ratio)
        f_cnt += filte_success
        cnt += len(bboxes)
        
        for bbox, score, label in zip(bboxes, scores, labels):
            result.append({
                'bbox': bbox.tolist(),
                'category_id': int(label),
                'image_id': int(image_id),
                'score': float(score)
            })
            
    if args.save_filter_results:
        out_file = os.path.join(args.out_dir, 'filter_boundary_boxes_results.json')
        dump(result, file=out_file)
        
        print_log(
            f'Filter results have been saved to {out_file}.', logger='current')
    
    print_log(
            f'Filter {f_cnt} boxes. Total {cnt} boxes, {f_cnt/cnt:.2%} boxes have been removed.', logger='current')

    

if __name__ == '__main__':
    main()