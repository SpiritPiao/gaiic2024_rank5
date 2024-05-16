import cv2
import json
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path

root = Path('/root/workspace/data/Visdrone')

'''
[
        {
            "id": 1,
            "name": "car"
        },
        {
            "id": 2,
            "name": "truck"
        },
        {
            "id": 3,
            "name": "bus"
        },
        {
            "id": 4,
            "name": "van"
        },
        {
            "id": 5,
            "name": "freight_car"
        }
    ]
'''

def make_coco(task: str = 'train'):
    dataset = {'images': [], 'annotations': [], 'categories': []}
    categories = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
    _categories = ['car', 'truck', 'bus', 'van', 'freight_car']
    # for i, cls in enumerate(categories):
    for i, cls in enumerate(_categories, 1):
        dataset['categories'].append({'id': i, 'name': cls})

    cnt = 0
    obj_count = 0
    for img in tqdm((root / task / 'rgb').iterdir()):
        im = cv2.imread(str(img))
        height, width = im.shape[:2]
        img_info_dict = {
            'file_name': img.name,
            'id': cnt,
            'width': width,
            'height': height
        }
        dataset['images'].append(img_info_dict)
        label = root / 'orin_text' / 'vs_labels' / task /(img.stem + '.txt')
        if not label.exists():
            label = root / 'orin_text' / 'vs_labels' / 'test-dev' /(img.stem + '.txt')
        with open(label, 'r') as f:
            for line in f:
                # *points, ignore, c, _, _  = line.rstrip().split(',')
                x1, y1, x2, y2, ignore, c, *_ = line.rstrip().split(',')
                if ignore == '0':
                    continue
                c = int(c) - 1
                _c = categories[c]
                if _c not in _categories:
                    continue
                c = _categories.index(_c) + 1
                # x1,y1,w,h
                points = np.array([x1, y1, x2, y2], dtype=np.int32)
                x1, y1, w, h = points.tolist()
                x1 = min(max(0, x1), width)
                y1 = min(max(0, y1), height)
                x2 = min(max(0, x1 + w), width)
                y2 = min(max(0, y1 + h), height)
                w = x2 - x1
                h = y2 - y1
                coco_format_info = {
                    'image_id': cnt,
                    'id': obj_count,
                    'category_id': c,
                    'bbox': [x1, y1, w, h],
                    'area': w * h,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
                    'iscrowd': 0
                }
                dataset['annotations'].append(coco_format_info)
                obj_count += 1

        cnt += 1

    with open(root / 'orin_text' /  (task + '.json'), 'w') as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    make_coco('train')
    make_coco('val')
