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

import torch

@TRANSFORMS.register_module()
class CachedMosaic2Images(Mosaic):
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

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop})'
        return repr_str



def radial_dark(image, centers, radius_list, angles):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for center, radius, angle in zip(centers, radius_list, angles):
        cv2.ellipse(mask, (int(center[1]), int(center[0])), (int(radius[1]), int(radius[0])), angle, 0, 360, 255, -1)

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    normalized_dist = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)[:, :, None]
    
    ratio = np.count_nonzero(normalized_dist > 0.5) / np.count_nonzero(normalized_dist >= 0.0)
    result = image * (1 - normalized_dist)

    return result.astype(np.uint8), ratio


@TRANSFORMS.register_module()
class RandDarkMask(BaseTransform):
    
    def __init__(self, prob=0.2,
                    dark_channel_prob=0.5, 
                    iteraiton=16,
                    random_position=True,
                    random_radius=True,
                    drak_channel_size=15,
                    radius_range=(36/1280, 256/1280), 
                    seed=0):
        self.iteraiton = iteraiton
        self.random_position = random_position
        self.random_radius = random_radius
        self.radius_range = radius_range
        self.prob = prob
        self.drak_channel_prob = dark_channel_prob
        self.drak_channel = DarkChannel(drak_channel_size)
        self.R = random.Random(seed)
    
    def _drak(self, image_3_channel, random_positions, random_radius, random_angles):
        # conver numpy HWC
        is_torch_tensor = False
        if isinstance(image_3_channel, torch.Tensor):
            is_torch_tensor = True
            image_3_channel = image_3_channel.cpu().numpy()
        # image_3_channel = image_3_channel.transpose((1, 2, 0))
        
        denormalize_centers = []
        denormalize_radius = []
        for p, r in zip(random_positions, random_radius):
            r0 = max(r[0] * image_3_channel.shape[0], 8)
            r1 = max(r[1] * image_3_channel.shape[1], 8)
            h = p[0] * image_3_channel.shape[0]
            w = p[1] * image_3_channel.shape[1]
            denormalize_centers.append([w, h])
            denormalize_radius.append([r0, r1])
        
        image_3_channel, ratio = radial_dark(image_3_channel, centers=denormalize_centers, radius_list=denormalize_radius, angles=random_angles)
        
        # conver numpy CHW
        # image_3_channel = image_3_channel.transpose((2, 0, 1))    
        
        if is_torch_tensor:
            image_3_channel = torch.from_numpy(image_3_channel)
        
        return image_3_channel, ratio
    
    @cache_randomness
    def random_parameters(self):
        random_positions = [[0.5, 0.5]]
        random_radius = [0.5]
        
        if self.random_position:
            random_positions = [[self.R.rand(), self.R.rand()] for _ in range(self.iteraiton)]
        if self.random_radius:
            min_, max_ = self.radius_range
            random_radius = [[self.R.rand() * (max_ - min_) + min_, self.R.rand() * (max_ - min_) + min_] for _ in range(self.iteraiton)]
        
        random_angles = [self.R.randint(0, 360) for _ in range(self.iteraiton)]
        
        return random_positions, random_radius, random_angles
    
    @cache_randomness
    def _random_prob(self) -> float:
        return self.R.uniform(0, 1)

    @cache_randomness
    def _random_prob_dark_channel(self) -> float:
        return self.R.uniform(0, 1)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        random_positions, random_radius, random_angles = self.random_parameters()
        if self._random_prob() < self.prob:
            img_shape = results['img'].shape[:2]
            img = results['img']
            if self._random_prob_dark_channel() >= self.drak_channel_prob:
                image_3_channel, ratio = self._drak(img, random_positions, random_radius, random_angles)
            else:
                res = self.drak_channel.run(img)
                image_3_channel = (res - res.min()) / (res.max() - res.min()) * 255
                
            results['img'] = image_3_channel

        return results


# DarkChannel
class DarkChannel:
    def __init__(self, sz=15):
        self._sz = sz

    def _dark_channel(self, im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self._sz, self._sz))
        dark = cv2.erode(dc, kernel)
        return dark

    def _atm_light(self, im, dark):
        [h, w] = im.shape[:2]
        imsz = h*w
        numpx = int(max(math.floor(imsz/1000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)

        indices = darkvec.argsort()
        indices = indices[imsz-numpx::]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def _transmission_estimate(self, im, A):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)

        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind]/A[0, ind]

        transmission = 1 - omega*self._dark_channel(im3)
        return transmission

    def _guided_filter(self, im, p, r, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I*mean_p

        mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I*mean_I

        a = cov_Ip/(var_I + eps)
        b = mean_p - a*mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a*im + mean_b
        return q

    def _transmission_refine(self, im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)/255
        r = 60  # default 60
        eps = 0.0001
        t = self._guided_filter(gray, et, r, eps)

        return t

    def _recover(self, im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)

        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind]-A[0, ind])/t + A[0, ind]

        return res

    def run(self, image):
        rever_img = 255 - image
        rever_img_bn = rever_img.astype('float64')/255
        dark = self._dark_channel(rever_img_bn)
        A = self._atm_light(rever_img_bn, dark)
        te = self._transmission_estimate(rever_img_bn, A)
        t = self._transmission_refine(image, te)
        J = self._recover(rever_img_bn, t, A, 0.1)

        rever_res_img = (1-J)*255
        return rever_res_img
    
    
if __name__ == "__main__":
    import imageio
    dc = DarkChannel()
    res = dc.run(cv2.imread("/root/workspace/data/GAIIC2024/val/rgb/00003.jpg"))
    # print(res.max(), res.min())
    res = (res - res.min()) / (res.max() - res.min()) * 220
    cv2.imwrite("debug3.png", res.astype(np.uint8))