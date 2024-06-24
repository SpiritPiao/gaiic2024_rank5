import cv2
import json
import numpy as np

from tqdm import tqdm
from pathlib import Path
from xml.etree import ElementTree as ET

root = Path('/root/workspace/data/DroneVehicle')
new_root = Path('/root/workspace/data/DroneVehicle/coco_format')
new_root.mkdir(parents=True, exist_ok=True)


def obj_to_coco_h(obj, categories: tuple, width: int, height: int, img_cnt: int, obj_cnt: int):
    cate = obj.find('name').text
    if 'feright' in cate:
        cate = 'freight_car'
    if cate == 'truvk':
        cate = 'truck'
    if cate not in categories:
        print(cate)
        return
    # pose = obj.find('pose').text
    # truncated = int(obj.find('truncated').text)
    # difficult = int(obj.find('difficult').text)
    box = obj.find('bndbox')
    if not box:
        return
    x1 = int(box.find('xmin').text) - 100
    y1 = int(box.find('ymin').text) - 100
    x2 = int(box.find('xmax').text) - 100
    y2 = int(box.find('ymax').text) - 100
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # cv2.rectangle(im, (x_left, y_top), (x_right, y_bottom), (255, 0, 0), 2)
    w = x2 - x1
    h = y2 - y1

    coco_format_info = {
        'image_id': img_cnt,
        'id': obj_cnt,
        'category_id': categories.index(cate) + 1,
        'bbox': [x1, y1, w, h],
        'area': w * h,
        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
        'iscrowd': 0
    }
    obj_cnt += 1
    return coco_format_info, obj_cnt


def obj_to_coco_r(obj, categories: tuple, width: int, height: int, img_cnt: int, obj_cnt: int):
    cate = obj.find('name').text
    if 'feright' in cate:
        cate = 'freight_car'
    if cate == 'truvk':
        cate = 'truck'
    if cate not in categories:
        print(cate)
        return
    # pose = obj.find('pose').text
    # truncated = int(obj.find('truncated').text)
    # difficult = int(obj.find('difficult').text)
    rbox = obj.find('polygon')
    if not rbox:
        return
    x1 = int(rbox.find('x1').text) - 100
    y1 = int(rbox.find('y1').text) - 100
    x2 = int(rbox.find('x2').text) - 100
    y2 = int(rbox.find('y2').text) - 100
    x3 = int(rbox.find('x3').text) - 100
    y3 = int(rbox.find('y3').text) - 100
    x4 = int(rbox.find('x4').text) - 100
    y4 = int(rbox.find('y4').text) - 100
    point_x = np.array([x1, x2, x3, x4], dtype=np.int32)
    point_y = np.array([y1, y2, y3, y4], dtype=np.int32)
    point_x = point_x.clip(min=0, max=width)
    point_y = point_y.clip(min=0, max=height)
    points = np.stack([point_x, point_y], axis=1)
    x_left = np.min(point_x).item()
    y_top = np.min(point_y).item()
    x_right = np.max(point_x).item()
    y_bottom = np.max(point_y).item()
    # cv2.rectangle(im, (x_left, y_top), (x_right, y_bottom), (255, 0, 0), 2)
    w = x_right - x_left
    h = y_bottom - y_top

    coco_format_info = {
        'image_id': img_cnt,
        'id': obj_cnt,
        'category_id': categories.index(cate) + 1,
        'bbox': [x_left, y_top, w, h],
        'area': w * h,
        'segmentation': [points.flatten().tolist()],
        'iscrowd': 0
    }
    obj_cnt += 1
    return coco_format_info, obj_cnt


def convert(mode: str = 'train'):
    dataset_rgb = {'images': [], 'annotations': [], 'categories': []}
    dataset_tir = {'images': [], 'annotations': [], 'categories': []}

    categories = ('car', 'truck', 'bus', 'van', 'freight_car')
    for i, cls in enumerate(categories, start=1):
        dataset_rgb['categories'].append({'id': i, 'name': cls})
        dataset_tir['categories'].append({'id': i, 'name': cls})

    path = root / mode
    img_save_path = new_root / 'images' / mode
    (img_save_path / 'rgb').mkdir(parents=True, exist_ok=True)
    (img_save_path / 'tir').mkdir(parents=True, exist_ok=True)
    img_cnt = 0
    obj_cnt_rgb = 0
    obj_cnt_tir = 0

    for i, img in tqdm(enumerate((path / f'{mode}img').iterdir())):
        bgr = cv2.imread(str(img))[100:-100, 100:-100, :]
        tir = cv2.imread(str(img.parent.parent / f'{mode}imgr' / img.name))[100:-100, 100:-100, :]
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        tir = np.ascontiguousarray(tir, dtype=np.uint8)
        # name = f'test-{img_cnt:04d}.jpg'
        name = f'test-{img.name}'
        cv2.imwrite(str(img_save_path / 'rgb' / name), bgr)
        cv2.imwrite(str(img_save_path / 'tir' / name), tir)
        assert bgr.shape == tir.shape
        height, width = bgr.shape[:2]
        img_info_dict = {
            'file_name': name,
            'id': img_cnt,
            'width': width,
            'height': height
        }
        dataset_rgb['images'].append(img_info_dict)
        dataset_tir['images'].append(img_info_dict)
        rgb_label = path / f'{mode}label' / (img.stem + '.xml')
        tir_label = path / f'{mode}labelr' / (img.stem + '.xml')

        rgb_root = ET.parse(rgb_label).getroot()
        tir_root = ET.parse(tir_label).getroot()

        for obj in rgb_root.iter('object'):
            ret_h = obj_to_coco_h(obj, categories, width, height, img_cnt, obj_cnt_rgb)
            if ret_h:
                coco_format_info, obj_cnt_rgb = ret_h
                dataset_rgb['annotations'].append(coco_format_info)
            ret_r = obj_to_coco_r(obj, categories, width, height, img_cnt, obj_cnt_rgb)
            if ret_r:
                coco_format_info, obj_cnt_rgb = ret_r
                dataset_rgb['annotations'].append(coco_format_info)

        for obj in tir_root.iter('object'):
            ret_h = obj_to_coco_h(obj, categories, width, height, img_cnt, obj_cnt_tir)
            if ret_h:
                coco_format_info, obj_cnt_tir = ret_h
                dataset_tir['annotations'].append(coco_format_info)
            ret_r = obj_to_coco_r(obj, categories, width, height, img_cnt, obj_cnt_tir)
            if ret_r:
                coco_format_info, obj_cnt_tir = ret_r
                dataset_tir['annotations'].append(coco_format_info)
        img_cnt += 1
    (new_root / 'annotations' / f'{mode}_rgb.json').parent.mkdir(parents=True, exist_ok=True)
    (new_root / 'annotations' / f'{mode}_tir.json').parent.mkdir(parents=True, exist_ok=True)
    with open(new_root / 'annotations' / f'{mode}_rgb.json', 'w') as f:
        json.dump(dataset_rgb, f)

    with open(new_root / 'annotations' / f'{mode}_tir.json', 'w') as f:
        json.dump(dataset_tir, f)


if __name__ == '__main__':
    convert('train')
    convert('val')
    convert('test')
