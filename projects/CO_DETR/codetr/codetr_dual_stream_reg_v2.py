import copy
import math
import time
from turtle import forward
from typing import Any, Mapping, Tuple, Union

import cv2
from mmengine.optim import OptimWrapper
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

from mmdet.models.layers.registration import SpatialTransformer
from projects.CO_DETR.codetr.registration_net import Unet
from torch.nn import functional as F

def ncc_loss(y_true, y_pred):

    I = y_true
    J = y_pred

    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -torch.mean(cc)


@MODELS.register_module()
class CoDETR_Dual_Reg_V2(BaseDetector):

    def __init__(
            self,
            backbone,
            neck=None,
            query_head=None,  # detr head
            rpn_head=None,  # two-stage rpn
            roi_head=[None],  # two-stage
            bbox_head=[None],  # one-stage
            train_cfg=[None, None],
            test_cfg=[None, None],
            # Control whether to consider positive samples
            # from the auxiliary head as additional positive queries.
            with_pos_coord=True,
            use_lsj=True,
            eval_module='detr',
            # Evaluate the Nth head.
            eval_index=0,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None):
        super(CoDETR_Dual_Reg_V2, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.with_pos_coord = with_pos_coord
        self.use_lsj = use_lsj

        assert eval_module in ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module

        self.backbone1 = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone)
        
        if neck is not None:
            self.neck = MODELS.build(neck)
        # Module index for evaluation
        self.eval_index = eval_index
        head_idx = 0
        if query_head is not None:
            query_head.update(train_cfg=train_cfg[head_idx] if (
                train_cfg is not None and train_cfg[head_idx] is not None
            ) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            self.query_head = MODELS.build(query_head)
            self.query_head.init_weights()
            head_idx += 1

        if rpn_head is not None:
            rpn_train_cfg = train_cfg[head_idx].rpn if (
                train_cfg is not None
                and train_cfg[head_idx] is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(
                train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
            self.rpn_head = MODELS.build(rpn_head_)
            self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                rcnn_train_cfg = train_cfg[i + head_idx].rcnn if (
                    train_cfg
                    and train_cfg[i + head_idx] is not None) else None
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg[i + head_idx].rcnn)
                self.roi_head.append(MODELS.build(roi_head[i]))
                self.roi_head[-1].init_weights()

        self.bbox_head = nn.ModuleList()
        for i in range(len(bbox_head)):
            if bbox_head[i]:
                bbox_head[i].update(
                    train_cfg=train_cfg[i + head_idx + len(self.roi_head)] if (
                        train_cfg and train_cfg[i + head_idx +
                                                len(self.roi_head)] is not None
                    ) else None)
                bbox_head[i].update(test_cfg=test_cfg[i + head_idx +
                                                      len(self.roi_head)])
                self.bbox_head.append(MODELS.build(bbox_head[i]))
                self.bbox_head[-1].init_weights()

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        intput_channels = 1
        self.reg_net = Unet(intput_channels * 2)
        self.spt = SpatialTransformer(size=(1024, 1024))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % 2)
        self.flow = Conv(2, 2, kernel_size=3, padding=1)

        from torch.distributions.normal import Normal

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Find backbone parameters
        copy_ori = False
        ori_backbone_params = []
        ori_backbone_key = []
        for k, v in state_dict.items():
            if k.startswith("backbone")  and "backbone1" not in k and "backbone2" not in k:
                # Pretrained on original model
                ori_backbone_params += [v]
                ori_backbone_key += [k]
                copy_ori = True
                
        if copy_ori:
            for k, v in zip(ori_backbone_key, ori_backbone_params):
                state_dict[k.replace("backbone", "backbone1")] = v
                state_dict[k.replace("backbone", "backbone2")] = copy.deepcopy(v)
                del state_dict[k]
            # Force set the strict to "False"
            strict = False
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


    def forward(self,
                    inputs: torch.Tensor,
                    inputs2: torch.Tensor,
                    data_samples: OptSampleList = None,
                    mode: str = 'tensor'):
            """The unified entry for a forward process in both training and test.

            The method should accept three modes: "tensor", "predict" and "loss":

            - "tensor": Forward the whole network and return tensor or tuple of
            tensor without any post-processing, same as a common nn.Module.
            - "predict": Forward and return the predictions, which are fully
            processed to a list of :obj:`DetDataSample`.
            - "loss": Forward and return a dict of losses according to the given
            inputs and data samples.

            Note that this method doesn't handle either back propagation or
            parameter update, which are supposed to be done in :meth:`train_step`.

            Args:
                inputs (torch.Tensor): The input tensor with shape
                    (N, C, ...) in general.
                data_samples (list[:obj:`DetDataSample`], optional): A batch of
                    data samples that contain annotations and predictions.
                    Defaults to None.
                mode (str): Return what kind of value. Defaults to 'tensor'.

            Returns:
                The return type depends on ``mode``.

                - If ``mode="tensor"``, return a tensor or a tuple of tensor.
                - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
                - If ``mode="loss"``, return a dict of tensor.
            """
            
            # from mmdet.visualization.local_visualizer import DetLocalVisualizer
            # dv = DetLocalVisualizer()
            # image = inputs2.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
            # image2 = inputs.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
            # dv.add_datasample('image', image, data_samples[0], draw_gt=True, show=True)
            # dv.add_datasample('image2', image2, data_samples[0], draw_gt=True, show=True)
            
            if mode == 'loss':
                return self.loss(inputs, inputs2, data_samples)
            elif mode == 'predict':
                return self.predict(inputs, inputs2, data_samples)
            elif mode == 'tensor':
                return self._forward(inputs, inputs2, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". '
                                'Only supports loss, predict and tensor mode')

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(
            self.roi_head) > 0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None
                 and len(self.roi_head) > 0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None
                    and len(self.bbox_head) > 0))

    def extract_feat(self, batch_inputs: Tensor, batch_inputs2: Tensor, output_flow=False, mode=None) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """

        ## Align Inputs
        
        # downsample_batch_inputs = F.interpolate(batch_inputs, size=[512, 512])
        # downsample_batch_inputs2 = F.interpolate(batch_inputs2, size=[512, 512])
        if output_flow:
            # Generate random affine matrix
            B = batch_inputs.size(0)
            input_shape = batch_inputs.shape
            
            rotation = (torch.rand(B, 1, 1) - 0.5) * torch.pi / 12
            rotation = torch.expand_copy(rotation, (B, 2, 2))

            rotation[:, 0, 0] = torch.cos(rotation[:, 0, 0])
            rotation[:, 1, 1] = torch.cos(rotation[:, 1, 1])
            rotation[:, 0, 1] = torch.sin(-rotation[:, 0, 1])
            rotation[:, 1, 0] = torch.sin(rotation[:, 1, 0])

            transpose = torch.clamp(torch.normal(mean=0, std=1, size=(B, 2, 1)) * 0.16, -0.2, 0.2)
            theta = torch.concat([rotation, transpose], dim=2) # B, 2, 3
            assert theta.shape == (B, 2, 3)

            grid = F.affine_grid(theta, input_shape, align_corners=True)\
                .to(dtype=batch_inputs.dtype, device=batch_inputs.device, non_blocking=True)
            
            # from mmdet.visualization.local_visualizer import DetLocalVisualizer
            # dv = DetLocalVisualizer()
            image_before = batch_inputs.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
            # image2 = inputs.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
            # dv.add_datasample('image', image, data_samples[0], draw_gt=True, show=True)
            # dv.add_datasample('image2', image2, data_samples[0], draw_gt=True, show=True)
            moving_batch_inputs = batch_inputs.detach()
            batch_inputs =  F.grid_sample(moving_batch_inputs, grid, align_corners=True, padding_mode="border")
            image_after = batch_inputs.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
            
        ori_size = [1024, 1024]
        downsample_inputs = F.interpolate(batch_inputs, [256, 256])
        downsample_moving_batch_inputs = F.interpolate(batch_inputs2, [256, 256])
        ds_flow = self.flow(self.reg_net(downsample_inputs[:, 0:1], downsample_moving_batch_inputs[:, 0:1]))
        flow = F.interpolate(ds_flow, ori_size, mode=self.spt.mode) * ori_size[0] / 256
        
        flow = (torch.sigmoid(flow) - 0.5) * 128 * 4
        print(flow.shape, flow.max(), flow.min())
        moved_batch_inputs = self.spt.forward(batch_inputs, flow)
        image_back = moved_batch_inputs.detach().permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
        if time.time_ns() % 8 == 0:
            cv2.imwrite("regnet_fixed.jpg", image_before)
            cv2.imwrite("regnet_moving.jpg", image_after)
            cv2.imwrite("regnet_moved.jpg", image_back)
        
        x = self.backbone1(moved_batch_inputs)
        y = x
        # y = self.backbone2(batch_inputs2)
        
        ## Concat ##
        z = [i + j for i, j in zip(x, y)]

        if self.with_neck:
            z = self.neck(z)
            
        if output_flow:
            return z, flow, moved_batch_inputs
        return z

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_inputs2: Tensor,
                 batch_data_samples: OptSampleList = None):
        pass

    # def forward():
    #     pass

    def loss(self, batch_inputs: Tensor, batch_inputs2: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        
        batch_input_shape = batch_data_samples[0].batch_input_shape
        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = batch_input_shape
                img_metas['img_shape'] = [input_img_h, input_img_w]

        x, flow, moved_batch_inputs = self.extract_feat(batch_inputs, batch_inputs2, output_flow=True, mode="train")
        # pred_grid = self.spt.get_locs(-flow)
        losses = dict()

        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses
        
        def flow_grad_loss(flow):
            new_losses2 = dict()
            dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
            dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
            new_losses2['flow_grad_loss'] = 10 * (torch.mean(dx) + torch.mean(dy))
            return new_losses2

        def sim_loss(moved_batch_inputs, batch_inputs):
            new_losses3 = dict()
            # diff = moved_batch_inputs - batch_inputs
            new_losses3['reg_sim_loss'] = 100 * ncc_loss(batch_inputs.mean(1, True), moved_batch_inputs.mean(1, True))
            return new_losses3

        def flow_l1_loss(moved_batch_inputs, batch_inputs):
            new_losses4 = dict()
            diff = moved_batch_inputs - batch_inputs
            new_losses4['flow_reg_loss'] = 100 * torch.mean(torch.abs(diff))
            return new_losses4

        losses.update(flow_grad_loss(flow))
        # losses.update(flow_l1_loss(gt_grid, pred_grid))
        losses.update(sim_loss(moved_batch_inputs, batch_inputs2))
        # print(losses)
        return losses
        # DETR encoder and decoder forward
        if self.with_query_head:
            bbox_losses, x = self.query_head.loss(x, batch_data_samples)
            losses.update(bbox_losses)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get(
                'rpn_proposal', self.test_cfg[self.head_idx].rpn)

            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, proposal_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)

            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)

            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            proposal_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].loss(x, proposal_list,
                                               batch_data_samples)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else:
                if 'pos_coords' in roi_losses.keys():
                    roi_losses.pop('pos_coords')
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)

        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].loss(x, batch_data_samples)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    bbox_losses.pop('pos_coords')
            bbox_losses = upd_loss(bbox_losses, idx=i + len(self.roi_head))
            losses.update(bbox_losses)

        if self.with_pos_coord and len(positive_coords) > 0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.loss_aux(x, positive_coords[i],
                                                       i, batch_data_samples)
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor, batch_inputs2: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']

        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = img_metas['batch_input_shape']
                img_metas['img_shape'] = [input_img_h, input_img_w]

        img_feats = self.extract_feat(batch_inputs, batch_inputs2, mode="predict")
        if self.with_bbox and self.eval_module == 'one-stage':
            results_list = self.predict_bbox_head(
                img_feats, batch_data_samples, rescale=rescale)
        elif self.with_roi_head and self.eval_module == 'two-stage':
            results_list = self.predict_roi_head(
                img_feats, batch_data_samples, rescale=rescale)
        else:
            results_list = self.predict_query_head(
                img_feats, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def predict_query_head(self,
                           mlvl_feats: Tuple[Tensor],
                           batch_data_samples: SampleList,
                           rescale: bool = True) -> InstanceList:
        return self.query_head.predict(
            mlvl_feats, batch_data_samples=batch_data_samples, rescale=rescale)

    def predict_roi_head(self,
                         mlvl_feats: Tuple[Tensor],
                         batch_data_samples: SampleList,
                         rescale: bool = True) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        if self.with_query_head:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            results = self.query_head.forward(mlvl_feats, batch_img_metas)
            mlvl_feats = results[-1]
        rpn_results_list = self.rpn_head.predict(
            mlvl_feats, batch_data_samples, rescale=False)
        return self.roi_head[self.eval_index].predict(
            mlvl_feats, rpn_results_list, batch_data_samples, rescale=rescale)

    def predict_bbox_head(self,
                          mlvl_feats: Tuple[Tensor],
                          batch_data_samples: SampleList,
                          rescale: bool = True) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        if self.with_query_head:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            results = self.query_head.forward(mlvl_feats, batch_img_metas)
            mlvl_feats = results[-1]
        return self.bbox_head[self.eval_index].predict(
            mlvl_feats, batch_data_samples, rescale=rescale)
