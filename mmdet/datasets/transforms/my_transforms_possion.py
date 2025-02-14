# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale
import albumentations as A

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

Number = Union[int, float]

from .transforms import Mosaic
number = 0

@TRANSFORMS.register_module()
class CopyPaste_Possion(BaseTransform):


    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 prob: float = 0.1) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob
        self.number_copy = 0
        # self.cropp_img = []
        # self.cropp_img_class = []
        # self.cropp_img_2 = []
    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        """Call function to collect indexes.

        Args:
            cache (list): The results cache.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # self.results_cache.append(copy.deepcopy(results))

        # if len(self.cropp_img) <= 4:
        #     return results


        # TODO: refactor mosaic to reuse these code.
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []



        mosaic_img = results['img']
        mosaic_img2 = results['img2']
            # adjust coordinate

        gt_bboxes_i = results['gt_bboxes']
        gt_bboxes_labels_i = results['gt_bboxes_labels']
        gt_ignore_flags_i = results['gt_ignore_flags']

        mosaic_bboxes.append(gt_bboxes_i)
        mosaic_bboxes_labels.append(gt_bboxes_labels_i)
        mosaic_ignore_flags.append(gt_ignore_flags_i)
        

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)
        if random.random() < self.prob:
            mosaic_img,mosaic_img2, mosaic_bboxes,mosaic_bboxes_labels,mosaic_ignore_flags = self.pro_copypaste_possion(mosaic_img.shape,mosaic_img, mosaic_img2,mosaic_bboxes,mosaic_bboxes_labels,mosaic_ignore_flags)


        # if self.bbox_clip_border:
        #     mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        # inside_inds = mosaic_bboxes.is_inside(
        #     [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        # mosaic_bboxes = mosaic_bboxes[inside_inds]
        # mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        # mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_2'] = mosaic_img2
        results['img_shape'] = mosaic_img.shape[:2]
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        return results
    def pro_copypaste_possion(self,img_size1, img4,img4_2, labels4, class4, mosaic_ignore_flags, mixup_number=12,
                                         change_rotate=True, change_size=True,copy_noBlack = True,
                                         new_scale=0.1, use_cache = False, max_cached_images=0,data_type = 'copy'):
        import random
        import cv2
        import torch
        cropp_img = []
        cropp_img_class = []
        cropp_img = []
        cropp_img_class = []
        cropp_img_2 = []

        for label, label1 in zip(labels4, class4):

            left = int(label.tensor.numpy()[0][0])
            lower = int(label.tensor.numpy()[0][1])
            right = int(label.tensor.numpy()[0][2])
            upper = int(label.tensor.numpy()[0][3])

            class_name = label1
            if left == right or upper == lower:
                continue
 
            cropped = img4[lower :upper , left  : right ]  # (left, upper, right, lower)
            cropped_2 = img4_2[lower :upper , left  : right ]  # (left, upper, right, lower)
            if copy_noBlack:
                threshold = 0.1
                gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                total_pixels = gray_image.size
                black_pixels = np.sum(gray_image < 15)
                black_percentage = black_pixels / total_pixels
                if black_percentage >= threshold:
                    # print("block")
                    continue
                # else:
                #     print(1)
            # cv2.imwrite("crop/crop_{}.png".format(str(self.number)), cropped)
            
            # cv2.imwrite("crop/crop_tir_{}.png".format(str(self.number)), cropped_2)
            # self.number += 1

            cropp_img.append(copy.deepcopy(cropped))
            cropp_img_2.append(copy.deepcopy(cropped_2))
            cropp_img_class.append(copy.deepcopy(class_name))
        # 判断标签丰富度
        # if len(labels4.tensor.numpy()) <= 20:
        #     mixup_number = 2 * mixup_number
        
        mixup_number = random.randint(2, 4)
        max_val = min(len(cropp_img) // 2, mixup_number)
        mixup_number = random.randint(0, max_val)
        if len(cropp_img) < mixup_number:
            mixup_number =  0
           

        for i in range(mixup_number):
            if use_cache:
                index = random.choices(range(0, len(self.cropp_img)), k=1)[0]
                im = self.cropp_img[index].copy()
                im_2 = self.cropp_img_2[index].copy()
                class_name = self.cropp_img_class[index].copy()
            else:
                index = random.choices(range(0, len(cropp_img)), k=1)[0]
                im = cropp_img[index].copy()
                im_2 = cropp_img_2[index].copy()
                class_name = cropp_img_class[index].copy()
            # if bigger:
            #     x_p = self.origin_p[index].copy()[0]
            #     y_p = self.origin_p[index].copy()[1]
            W = im.shape[1]
            H = im.shape[0]
            if change_size:
                # if class_name == 1:
                #     scale = random.uniform(1, 1.1)
                # else:
                scale = random.uniform(1 - new_scale, 1 + new_scale)
                W = int(W * scale)
                H = int(H * scale)
                # x_p = int(x_p * scale)
                # y_p = int(y_p * scale)

                if H == 0 or W == 0:
                    continue
                im = cv2.resize(im, (W, H))
                im_2 = cv2.resize(im_2, (W, H))

            if len(im):
                if change_rotate:
                    # 随机旋转
                    if random.random() < 0.25:
                        im = np.rot90(im, -1)
                        im_2 = np.rot90(im_2, -1)

                    elif random.random() < 0.25:
                        im = np.rot90(im, 1)
                        im_2 = np.rot90(im_2, 1)

                    elif random.random() < 0.25:
                        im = np.rot90(im, 2)
                        im_2 = np.rot90(im_2, 2)
                    else:
                        pass
                    W = im.shape[1]
                    H = im.shape[0]

                point = [random.randint(20, img4.shape[1] - W -20), random.randint(20, img4.shape[0] -H- 20)]

                if data_type == 'copy':
                
                    img4[point[1]:point[1] + H, point[0]:point[0] + W] = im
                    img4_2[point[1]:point[1] + H, point[0]:point[0] + W] = im_2
                elif data_type == 'mixup':
                    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
                    im = (im * r + img4[point[1]:point[1] + H, point[0]:point[0] + W] * (1 - r)).astype(np.uint8)
                    img4[point[1]:point[1] + H, point[0]:point[0] + W] = im
                elif data_type == 'possion':
                    x_center = point[0] + W // 2
                    y_center = point[1] + H // 2
                    mask = 255 * np.ones(im.shape, im.dtype)
                    # mask_feathered = cv2.GaussianBlur(mask, (15, 15), 0)
                    try:
                        img4 = cv2.seamlessClone(im, img4, mask, (x_center, y_center), 0)
                        img4_2 = cv2.seamlessClone(im_2, img4_2, mask, (x_center, y_center), 0)
                    except:
                        print('possion fail')
                        img4[point[1]:point[1] + H, point[0]:point[0] + W] = im
                        img4_2[point[1]:point[1] + H, point[0]:point[0] + W] = im_2

                labels_single_img = np.array([[point[0], point[1], point[0] + W, point[1] + H]],
                                                 dtype=np.float32)
                # r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
                # img4 = (img4 * r + img4_2 * (1 - r)).astype(np.uint8)
                # cv2.rectangle(img4, (point[0], point[1]), (point[0] + W, point[1] + H),
                #               (0, 0, 0), 2)
                # cv2.putText(img4, str(class_name), (point[0], point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (0, 0, 0), 2)
                # cv2.rectangle(img4_2, (point[0], point[1]), (point[0] + W, point[1] + H),
                #               (0, 0, 0), 2)
                # cv2.putText(img4_2, str(class_name), (point[0], point[1] ), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (0, 0, 0), 2)
                # cv2.imwrite("copy/copy_{}.png".format(str(self.number_copy)), img4)
                # print("save")
        
                # cv2.imwrite("copy/copy_tir_{}.png".format(str(self.number_copy)), img4_2)
                # self.number_copy += 1
                
                labels4.tensor = torch.cat((labels4.tensor, torch.tensor(labels_single_img)), 0)

                class4 = np.append(class4, class_name)
                mosaic_ignore_flags = np.append(mosaic_ignore_flags, 0)
        # if use_cache and len(self.cropp_img) >= max_cached_images:
        #     index1 = random.choices(range(0, len(self.cropp_img)), k=max_cached_images // 2)
        #     counter = 0
        #     for index_del in index1:
        #         index_del = index_del - counter
        #         self.cropp_img.pop(index_del)
        #         self.cropp_img_2.pop(index_del)
        #         self.cropp_img_class.pop(index_del)
        #         # self.origin_p.pop(index_del)
        #         counter += 1
        return img4, img4_2, labels4, class4, mosaic_ignore_flags



    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
    
@TRANSFORMS.register_module()
class Pre_Pianyi(BaseTransform):


    def __init__(self,
                 canvas_size: Tuple[int, int] = (640, 640),
                 p: float = 1.0,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob
        self.number = 0
        self.canvas_size = canvas_size
        self.p = p
    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        # print(1)

        if  random.random() < self.p:
            # self.canvas_size = (660, 532)
            canvas = np.full((self.canvas_size[1], self.canvas_size[0], 3),114.0,dtype=np.uint8)
            img1 = results['img']
            image_size = img1.shape[:2]  
            max_left = self.canvas_size[0] - image_size[1]  
            max_top = self.canvas_size[1] - image_size[0]  
            random_left = random.randint(0, max_left)
            random_top = random.randint(0, max_top)
            canvas[random_top:random_top+image_size[0], random_left:random_left+image_size[1]] = img1
            crop_size = (image_size[1]  , image_size[0])

            x_min = random.randint(0, self.canvas_size[0] - crop_size[0])  
            y_min = random.randint(0, self.canvas_size[1] - crop_size[1])  
            cropped_image = canvas[y_min:y_min+crop_size[1], x_min:x_min+crop_size[0]]
            # print(1)
            
            results['img'] = cropped_image

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str

    
@TRANSFORMS.register_module()
class Pre_Pianyi_Bili(BaseTransform):


    def __init__(self,
                 canvas_size: Tuple[int, int] = (640, 640),
                 p: float = 1.0,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob
        self.number = 0
        self.canvas_size = canvas_size
        self.p = p
    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        # print(1)

        if  random.random() < self.p:
            
            img1 = results['img']
            image_size = img1.shape[:2] 
            if results['img_shape'][0] == 640 or results['img_shape'][1] == 640:
                bili = 1.046875
            else:
                bili = 1.01
                # print(results['img_shape'][0])
                # print(results['img_shape'][1])
            self.canvas_size = [0,0]
            self.canvas_size[0] = int(image_size[1]  * bili)
            self.canvas_size[1] = int(image_size[0]  * bili)
            canvas = np.full((self.canvas_size[1], self.canvas_size[0], 3),114.0,dtype=np.uint8)
            max_left = self.canvas_size[0] - image_size[1]  
            max_top = self.canvas_size[1] - image_size[0]  
            random_left = random.randint(0, max_left)
            random_top = random.randint(0, max_top)
            canvas[random_top:random_top+image_size[0], random_left:random_left+image_size[1]] = img1
            crop_size = (image_size[1]  , image_size[0])

            x_min = random.randint(0, self.canvas_size[0] - crop_size[0])  
            y_min = random.randint(0, self.canvas_size[1] - crop_size[1])  
            cropped_image = canvas[y_min:y_min+crop_size[1], x_min:x_min+crop_size[0]]
            # print(1)
            
            results['img'] = cropped_image

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
    
@TRANSFORMS.register_module()
class Get_three_mixup(BaseTransform):


    def __init__(self,
                 canvas_size: Tuple[int, int] = (640, 640),
                 p: float = 1.0,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob
        self.number = 0
        self.canvas_size = canvas_size
        self.p = p
    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        # print(1)
        img1 = results['img'] 
        img2 = results['img2'] 


        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        mixup_img = (img2 * r + img1 * (1 - r)).astype(np.uint8)
        results['img3'] = mixup_img
        results['img_shape3'] = mixup_img.shape[:2]
        results['ori_shape3'] = mixup_img.shape[:2]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
    
@TRANSFORMS.register_module()
class BBox_Jitter(BaseTransform):


    def __init__(self,
                prob: float = 0.5,
                max_shift_px: int = 2,
                filter_thr_px: int = 1,
                unchange_thr_px: int = 200) -> None:
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        assert max_shift_px >= 0
        self.prob = prob
        self.max_shift_px = max_shift_px
        self.filter_thr_px = int(filter_thr_px)
        self.unchange_thr_px = int(unchange_thr_px)
    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:


        gt_bboxes = results['gt_bboxes']
        img_shape = results['img'].shape[:2]
        for i in range(len(gt_bboxes)):
            gt_bbox = gt_bboxes[i].clone()

            if (gt_bbox.cxcywh[0, 2:]).min() > self.unchange_thr_px and random.random() < self.prob:
                random_shift_x = random.randint(-self.max_shift_px - 1,
                                                self.max_shift_px)
                random_shift_y = random.randint(-self.max_shift_px - 1,
                                                self.max_shift_px)
                gt_bbox.translate_([random_shift_x, random_shift_y])

                # clip border
                gt_bbox.clip_(img_shape)

                if gt_bbox.cxcywh[0, 2:].min() > self.filter_thr_px:
                    gt_bboxes[i] = gt_bbox

    
            
        results['gt_bboxes'] = gt_bboxes

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str

    
# @TRANSFORMS.register_module()
# class BBox_Jitter(BaseTransform):


#     def __init__(self,
#                 prob: float = 0.5,
#                 max_shift_px: int = 4,
#                 filter_thr_px: int = 1,
#                 unchange_thr_px: int = 200) -> None:
#         assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
#                                  f'got {prob}.'

#         assert max_shift_px >= 0
#         self.prob = prob
#         self.max_shift_px = max_shift_px
#         self.filter_thr_px = int(filter_thr_px)
#         self.unchange_thr_px = int(unchange_thr_px)
#     @cache_randomness
#     def get_indexes(self, cache: list) -> list:
#         indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
#         return indexes

#     @autocast_box_type()
#     def transform(self, results: dict) -> dict:


#         gt_bboxes = results['gt_bboxes']
#         img_shape = results['img'].shape[:2]
#         for i in range(len(gt_bboxes)):
#             gt_bbox = gt_bboxes[i].clone()

#             if (gt_bbox.cxcywh[0, 2:]).min() > self.unchange_thr_px and random.random() < self.prob:
#                 random_shift_x = random.randint(-self.max_shift_px,
#                                                 self.max_shift_px)
#                 random_shift_y = random.randint(-self.max_shift_px,
#                                                 self.max_shift_px)
#                 gt_bbox.translate_([random_shift_x, random_shift_y])

#                 # clip border
#                 gt_bbox.clip_(img_shape)

#                 if gt_bbox.cxcywh[0, 2:].min() > self.filter_thr_px:
#                     gt_bboxes[i] = gt_bbox

    
            
#         results['gt_bboxes'] = gt_bboxes

#         return results


#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(img_scale={self.img_scale}, '
#         repr_str += f'center_ratio_range={self.center_ratio_range}, '
#         repr_str += f'pad_val={self.pad_val}, '
#         repr_str += f'prob={self.prob})'
#         return repr_str

@TRANSFORMS.register_module()
class Albumentation(BaseTransform):


    def __init__(self,
                prob: float = 1) -> None:
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
 
        self.prob = prob

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img1 = results['img']
        img2 = results['img2']


        T = [
                A.Blur(p=0.03),
                A.MedianBlur(p=0.03),
                A.MotionBlur(p=0.03),
                A.RandomBrightnessContrast(p=0.03),
                A.ImageCompression(quality_lower = 75, p = 0.03),
                # A.RGBShift(p = 0.02),
                # A.CLAHE(p = 0.02)


                ]  # transforms
        albu_tr = albumentations.Compose(T)

        if random.random() < self.prob:
            new1 = albu_tr(image=img1)['image']
            new2 = albu_tr(image=img2)['image']
            img1 = new1
            img2 = new2

    
            
        results['img'] = img1
        results['img2'] = img2

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
@TRANSFORMS.register_module()
class CLAHE(BaseTransform):


    def __init__(self,
                prob: float = 1,
                size :int =8
                ) -> None:
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
 
        self.prob = prob
        self.size =size

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        
        if random.random() < self.prob:
            img2 = results['img2']
            channels = cv2.split(img2)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(self.size, self.size))
            clahe_channels = [clahe.apply(channel) for channel in channels]
            clahe_image = cv2.merge(clahe_channels)

            results['img2'] = clahe_image

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
    
@TRANSFORMS.register_module()
class Bright(BaseTransform):


    def __init__(self,
                prob: float = 1,
                ) -> None:
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
 
        self.prob = prob

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes
    def is_dark_image(image, threshold=50):
        assert len(image.shape) in [2, 3], "Invalid input image!"
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray_image)
        return mean_brightness < threshold

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        
        if random.random() < self.prob:
            img1 = results['img']
            # gray_image = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            image_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray_image)
            is_dark =  mean_brightness < 50

            if is_dark:
                image_float = image_rgb.astype(np.float32) / 255.0

                alpha = 2.0  # 提亮系数，可以根据需要调整
                bright_image = np.clip(image_float * alpha, 0, 1)
                bright_image_uint8 = (bright_image * 255).astype(np.uint8)
                img1 = bright_image_uint8
            else:
                pass

            results['img'] = img1

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
    
@TRANSFORMS.register_module()
class RandomCLAHE(BaseTransform):


    def __init__(self,
                prob: float = 1,
                size :int =8
                ) -> None:
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
 
        self.prob = prob
        self.size =size

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        size = [8, 8, 8, 16, 32, 64]
        chosen_size = random.choice(size)
        p_no = 0.1
        if random.random() < 1 - p_no:
            img2 = results['img2']
            channels = cv2.split(img2)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(chosen_size, chosen_size))
            clahe_channels = [clahe.apply(channel) for channel in channels]
            clahe_image = cv2.merge(clahe_channels)

            results['img2'] = clahe_image
        else:
            pass

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
    
@TRANSFORMS.register_module()
class Cache_Mixup(BaseTransform):


    def __init__(self,
                 prob: float = 1.0,
                 max_cache = 100) -> None:
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'

        self.prob = prob
        self.max_cache = max_cache
        self.mixup_img_rgb = []
        self.mixup_img_tir = []
        self.gt_bboxes = []
        self.gt_bboxes_labels = []
        self.gt_ignore_flags = []
        self.number = 0
    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        """Call function to collect indexes.

        Args:
            cache (list): The results cache.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        import random
        if random.random() < self.prob:
            img1 = results['img']
            img2 = results['img2']
            gt_bboxes = results['gt_bboxes']
            gt_bboxes_labels = results['gt_bboxes_labels']
            gt_ignore_flags = results['gt_ignore_flags']

            img_shape = results['img_shape']
            if img_shape[1] == 640:

                self.mixup_img_rgb.append(copy.deepcopy(img1))
                self.mixup_img_tir.append(copy.deepcopy(img2))
                self.gt_bboxes.append(copy.deepcopy(gt_bboxes))
                self.gt_bboxes_labels.append(copy.deepcopy(gt_bboxes_labels))
                self.gt_ignore_flags.append(copy.deepcopy(gt_ignore_flags))
                if len(self.mixup_img_rgb) >= 1:

                    index = random.choices(range(0, len(self.mixup_img_rgb)), k=1)[0]
                    new_mixup_img_rgb = self.mixup_img_rgb[index].copy()
                    new_mixup_img_tir = self.mixup_img_tir[index].copy()
                    new_gt_bboxes = self.gt_bboxes[index]
                    new_gt_bboxes_labels = self.gt_bboxes_labels[index].copy()
                    new_gt_ignore_flags = self.gt_ignore_flags[index].copy()

                    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
                    out_mixup_img_rgb = (img1 * r + new_mixup_img_rgb * (1 - r)).astype(np.uint8)
                    out_mixup_img_tir = (img2 * r + new_mixup_img_tir * (1 - r)).astype(np.uint8)
                    out_gt_bboxes = gt_bboxes.cat(
                                    (new_gt_bboxes, gt_bboxes), dim=0)
                    out_gt_bboxes_labels = np.concatenate(
                                    (gt_bboxes_labels, new_gt_bboxes_labels), axis=0)
                    mixup_gt_ignore_flags = np.concatenate(
                                    (gt_ignore_flags, new_gt_ignore_flags), axis=0)
                    
                    results['img'] = out_mixup_img_rgb
                    results['img2'] = out_mixup_img_tir
                    results['gt_bboxes'] = out_gt_bboxes
                    results['gt_bboxes_labels'] = out_gt_bboxes_labels
                    results['gt_ignore_flags'] = mixup_gt_ignore_flags
                else:
                    pass

                if len(self.mixup_img_rgb) >= self.max_cache:
                    index1 = random.choices(range(0, len(self.mixup_img_rgb)), k=self.max_cache // 2)
                    counter = 0
                    for index_del in index1:
                        index_del = index_del - counter
                        self.mixup_img_rgb.pop(index_del)
                        self.mixup_img_tir.pop(index_del)
                        self.gt_bboxes.pop(index_del)
                        self.gt_bboxes_labels.pop(index_del)
                        self.gt_ignore_flags.pop(index_del)
                        counter += 1
                
        return results
   



    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@TRANSFORMS.register_module()
class CachedMosaic2Images_Possion(Mosaic):
    """Cached mosaic augmentation.

    Cached mosaic transform will random select images from the cache
    and combine them into one output image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The cached mosaic transform steps are as follows:

         1. Append the results from the last transform into the cache.
         2. Choose the mosaic center as the intersections of 4 images
         3. Get the left top image according to the index, and randomly
            sample another 3 images from the result cache.
         4. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (np.float32) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
    """

    def __init__(self,
                 *args,
                 max_cached_images: int = 40,
                 random_pop: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.results_cache = []
        self.random_pop = random_pop
        assert max_cached_images >= 4, 'The length of cache must >= 4, ' \
                                       f'but got {max_cached_images}.'
        self.max_cached_images = max_cached_images
        self.cropp_img = []
        self.cropp_img_class = []
        self.origin_p = []

        self.cropp_img_2 = []
        self.cropp_img_class = []
        self.origin_p = []

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        """Call function to collect indexes.

        Args:
            cache (list): The results cache.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return results

        if random.uniform(0, 1) > self.prob:
            return results
        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO: refactor mosaic to reuse these code.
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        with_mask = True if 'gt_masks' in results else False

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
            mosaic_img2 = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img2'].dtype)
            
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)
            mosaic_img2 = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img2'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(mix_results[i - 1])

            img_i = results_patch['img']
            img_i2 = results_patch['img2']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
            
            
            img_i2 = mmcv.imresize(
                img_i2, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
                        # compute the combine parameters
            paste_coord2, crop_coord2 = self._mosaic_combine(
                loc, center_position, img_i2.shape[:2][::-1])
            
            x1_p, y1_p, x2_p, y2_p = paste_coord2
            x1_c, y1_c, x2_c, y2_c = crop_coord2
            mosaic_img2[y1_p:y2_p, x1_p:x2_p] = img_i2[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)



            if with_mask and results_patch.get('gt_masks', None) is not None:
                gt_masks_i = results_patch['gt_masks']
                gt_masks_i = gt_masks_i.rescale(float(scale_ratio_i))
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction='horizontal')
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction='vertical')
                mosaic_masks.append(gt_masks_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        mosaic_img,mosaic_img2, mosaic_bboxes,mosaic_bboxes_labels,mosaic_ignore_flags = self.pro_copypaste_possion(mosaic_img.shape,mosaic_img, mosaic_img2,mosaic_bboxes,mosaic_bboxes_labels,mosaic_ignore_flags)


        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img2'] = mosaic_img2
        results['img_shape'] = mosaic_img.shape[:2]
        results['img_shape2'] = mosaic_img2.shape[:2]
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        if with_mask:
            mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
            results['gt_masks'] = mosaic_masks[inside_inds]
        return results
    def pro_copypaste_possion(self,img_size1, img4,img4_2, labels4, class4, mosaic_ignore_flags, mixup_number=12,
                                         change_rotate=True, change_size=True,
                                         new_scale=0.1, use_cache = True, max_cached_images=100,data_type = 'possion'):
        import random
        import cv2
        import torch
        cropp_img = []
        cropp_img_class = []

        for label, label1 in zip(labels4, class4):

            left = int(label.tensor.numpy()[0][0])
            lower = int(label.tensor.numpy()[0][1])
            right = int(label.tensor.numpy()[0][2])
            upper = int(label.tensor.numpy()[0][3])
            # if bigger:
            #     beilv_p = random.uniform(0, 0.3)
            #     H_p = (upper - lower) * beilv_p
            #     W_p = (right - left) * beilv_p
            #     x_p = int(W_p // 2)
            #     y_p = int(H_p // 2)
            # else:
            #     x_p = 0
            #     y_p = 0

            class_name = label1
            if left == right or upper == lower:
                continue
            # cropped = img4[lower - y_p:upper + y_p, left - x_p : right + x_p]  # (left, upper, right, lower)
            # cropped_2 = img4_2[lower - y_p:upper + y_p, left - x_p : right + x_p]  # (left, upper, right, lower)
            cropped = img4[lower :upper , left  : right ]  # (left, upper, right, lower)
            cropped_2 = img4_2[lower :upper , left  : right ]  # (left, upper, right, lower)
            if use_cache:
                self.cropp_img.append(copy.deepcopy(cropped))
                self.cropp_img_2.append(copy.deepcopy(cropped_2))
                self.cropp_img_class.append(copy.deepcopy(class_name))
                # self.origin_p.append([x_p, y_p])
            else:
                cropp_img.append(copy.deepcopy(cropped))
                self.cropp_img_2.append(copy.deepcopy(cropped_2))
                self.cropp_img_class.append(copy.deepcopy(class_name))
        # 判断标签丰富度
        # if len(labels4.tensor.numpy()) <= 20:
        #     mixup_number = 2 * mixup_number
        if use_cache:
            mixup_number = random.randint(1, 8)
        else:
            max_val = min(len(cropp_img) // 2, mixup_number)
            mixup_number = random.randint(0, max_val)

        for i in range(mixup_number):
            if use_cache:
                index = random.choices(range(0, len(self.cropp_img)), k=1)[0]
                im = self.cropp_img[index].copy()
                im_2 = self.cropp_img_2[index].copy()
                class_name = self.cropp_img_class[index].copy()
            else:
                index = random.choices(range(0, len(cropp_img)), k=1)[0]
                im = cropp_img[index].copy()
                im_2 = self.cropp_img_2[index].copy()
                class_name = cropp_img_class[index].copy()
            # if bigger:
            #     x_p = self.origin_p[index].copy()[0]
            #     y_p = self.origin_p[index].copy()[1]
            W = im.shape[1]
            H = im.shape[0]
            if change_size:
                # if class_name == 1:
                #     scale = random.uniform(1, 1.1)
                # else:
                scale = random.uniform(1 - new_scale, 1 + new_scale)
                W = int(W * scale)
                H = int(H * scale)
                # x_p = int(x_p * scale)
                # y_p = int(y_p * scale)

                if H == 0 or W == 0:
                    continue
                im = cv2.resize(im, (W, H))
                im_2 = cv2.resize(im, (W, H))

            if len(im):
                if change_rotate:
                    # 随机旋转
                    if random.random() < 0.25:
                        im = np.rot90(im, -1)
                        im_2 = np.rot90(im_2, -1)
                        # if bigger:
                        #     temp = x_p
                        #     x_p = y_p
                        #     y_p = temp
                    elif random.random() < 0.25:
                        im = np.rot90(im, 1)
                        im_2 = np.rot90(im_2, 1)
                        # if bigger:
                        #     temp = x_p
                        #     x_p = y_p
                        #     y_p = temp
                    elif random.random() < 0.25:
                        im = np.rot90(im, 2)
                        im_2 = np.rot90(im_2, 2)
                    else:
                        pass
                    W = im.shape[1]
                    H = im.shape[0]
                    # W_2 = im_2.shape[1]
                    # H_2 = im_2.shape[0]
                    # print(W, H)
                    # print(W_2, H_2)
                point = [random.randint(50 + W, img_size1[1] - W - 50), random.randint(50, img_size1[0] - H - 50)]
                if data_type == 'copy':
                    img4[point[1]:point[1] + H, point[0]:point[0] + W] = im
                elif data_type == 'mixup':
                    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
                    im = (im * r + img4[point[1]:point[1] + H, point[0]:point[0] + W] * (1 - r)).astype(np.uint8)
                    img4[point[1]:point[1] + H, point[0]:point[0] + W] = im
                elif data_type == 'possion':
                    x_center = point[0] + W // 2
                    y_center = point[1] + H // 2
                    mask = 255 * np.ones(im.shape, im.dtype)
                    # mask_feathered = cv2.GaussianBlur(mask, (15, 15), 0)
                    try:
                        img4 = cv2.seamlessClone(im, img4, mask, (x_center, y_center), 0)
                        img4_2 = cv2.seamlessClone(im_2, img4_2, mask, (x_center, y_center), 0)
                    except:
                        print('possion fail')
                        img4[point[1]:point[1] + H, point[0]:point[0] + W] = im
                        img4_2[point[1]:point[1] + H, point[0]:point[0] + W] = im_2
                        # r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
                        # im = (im * r + img4[point[1]:point[1] + H, point[0]:point[0] + W] * (1 - r)).astype(np.uint8)
                        # img4[point[1]:point[1] + H, point[0]:point[0] + W] = im

                # if bigger:
                #     labels_single_img = np.array([[point[0] + x_p, point[1]  + y_p, point[0] + W  - x_p, point[1] + H  - y_p]],
                #                                  dtype=np.float32)
                # else:
                labels_single_img = np.array([[point[0], point[1], point[0] + W, point[1] + H]],
                                                 dtype=np.float32)
                
                # cv2.rectangle(img4, (point[0] + x_p, point[1] + y_p), (point[0] + W - x_p, point[1] + H - y_p),
                #               (0, 0, 0), 2)
                # cv2.putText(img4, str(classesname[class_name]), (point[0]+ x_p, point[1] + y_p), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (0, 0, 0), 2)
                #
                # cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("Demo", 1280, 800)
                # cv2.imshow("Demo", img4)
                # cv2.waitKey(0)  # 等待用户按键触发
                # cv2.imwrite("1.png", img4)
                # labels4_numpy = np.concatenate((labels_single_img, labels4.tensor.numpy()), 0)
                labels4.tensor = torch.cat((labels4.tensor, torch.tensor(labels_single_img)), 0)
                class4 = np.append(class4, class_name)
                mosaic_ignore_flags = np.append(mosaic_ignore_flags, 0)
        if use_cache and len(self.cropp_img) >= max_cached_images:
            index1 = random.choices(range(0, len(self.cropp_img)), k=max_cached_images // 2)
            counter = 0
            for index_del in index1:
                index_del = index_del - counter
                self.cropp_img.pop(index_del)
                self.cropp_img_2.pop(index_del)
                self.cropp_img_class.pop(index_del)
                # self.origin_p.pop(index_del)
                counter += 1
        return img4, img4_2, labels4, class4, mosaic_ignore_flags

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop})'
        return repr_str
    
