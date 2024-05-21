import copy
from turtle import forward
from typing import Any, Mapping, Tuple, Union

import cv2
from mmengine.optim import OptimWrapper
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
# from .dual_resnet import Dual_ResNet

@MODELS.register_module()
class CoDETR_Dual_Swin_Vat(BaseDetector):

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
        super(CoDETR_Dual_Swin_Vat, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.with_pos_coord = with_pos_coord
        self.use_lsj = use_lsj

        assert eval_module in ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module

        self.backbone1 = MODELS.build(backbone)
        # self.backbone2 = MODELS.build(backbone)
        
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
        self.kl_loss = nn.KLDivLoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")


        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Find backbone parameters
        copy_ori = False
        ori_backbone_params = []
        ori_backbone_key = []
        for k, v in state_dict.items():
            # if k.startswith("backbone.layer1")  and "layer1_t" not in k:
            if ("backbone2.patch_embed" in k and "backbone1.patch_embed1" not in k) or  ("backbone2.stages" in k and "backbone1.stages1" not in k
            or ("backbone2.norm" in k and "backbone1.tir_norm"  not in k) ):
                # Pretrained on original model
                ori_backbone_params += [v]
                ori_backbone_key += [k]
                copy_ori = True
            # if ("backbone.patch_embed" in k and "backbone.patch_embed1" not in k) or  ("backbone.stages" in k and "backbone.stages1" not in k
            # or ("backbone.norm" in k and "backbone.tir_norm"  not in k) ):
            #     # Pretrained on original model
            #     ori_backbone_params += [v]
            #     ori_backbone_key += [k]
            #     copy_ori = False
                
        if copy_ori:
            for k, v in zip(ori_backbone_key, ori_backbone_params):
                state_dict[k.replace("backbone2.patch_embed", "backbone1.patch_embed1")] = copy.deepcopy(v)
                state_dict[k.replace("backbone2.stages", "backbone1.stages1")] = copy.deepcopy(v)
                state_dict[k.replace("backbone2.norm" , "backbone1.tir_norm")] = copy.deepcopy(v)
            
                # state_dict[k] = v
                # state_dict[k.replace("backbone.patch_embed", "backbone.patch_embed1")] = copy.deepcopy(v)
                # state_dict[k.replace("backbone.stages", "backbone.stages1")] = copy.deepcopy(v)
                # state_dict[k.replace("backbone.norm" , "backbone.tir_norm")] = copy.deepcopy(v)
                # del state_dict[k]
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

    def extract_feat(self, batch_inputs: Tensor, batch_inputs2: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
                
        # x = list(self.backbone1(batch_inputs))
        # y = list(self.backbone2(batch_inputs2))
        z = list(self.backbone1([batch_inputs,batch_inputs2]))
        
        ## Concat #
        # x[0], y[0] = self.eaef1([x[0], y[0]])
        # x[1], y[1] = self.eaef2([x[1], y[1]])
        # x[2], y[2] = self.eaef3([x[2], y[2]])
        # x[3], y[3] = self.eaef4([x[3], y[3]])
        # z = [i + j for i, j in zip(x, y)]
        
        if self.with_neck:
            z = self.neck(z)
        return z

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_inputs2: Tensor,
                 batch_data_samples: OptSampleList = None):
        pass

    def noise(self, shape, device_, eps=1e-3, std=1):
        delta_ = torch.normal(torch.zeros(shape, requires_grad=False), std=std).to(device_) * eps
        return delta_

    def add_noise(self, feats):
        feats_w_noise = []
        feats_wo_noise = [i.detach() for i in feats]
        for feat in feats:
            feats_w_noise.append(self.noise(feat.shape, feat.device,  torch.abs(feat.max() - feat.min()) * 5e-4) + feat.detach())
        return feats_wo_noise, feats_w_noise

    def query_head_forward(self, batch_data_samples):
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)

        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.query_head.dn_generator(batch_data_samples)

        return batch_img_metas, dn_label_query, dn_bbox_query, attn_mask
        
    def loss(self, batch_inputs: Tensor, batch_inputs2: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        
        batch_input_shape = batch_data_samples[0].batch_input_shape
        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = batch_input_shape
                img_metas['img_shape'] = [input_img_h, input_img_w]

        x = self.extract_feat(batch_inputs, batch_inputs2)

        # Add noise
        x_, x_noise = self.add_noise(x)
        
        losses = dict()
        
        
        def kl_div_adv(flatten_pl, flatten_pln):
            return self.kl_loss(torch.log_softmax(flatten_pl, -1), torch.softmax(flatten_pln.detach(), 1)) +  self.kl_loss(torch.log_softmax(flatten_pln, -1), torch.softmax(flatten_pl.detach(), 1))


        def consistent_loss(cls_, cls_w_noise, coords_, coords_w_noise):
            new_losses = dict()
            # cls_ = torch.concat(cls_list, dim=1)
            # cls_w_noise = torch.concat(cls_w_noise_list, dim=1)
            # coords_ = torch.concat(coords_list, dim=1)
            # coords_w_noise = torch.concat(coords_w_noise_list, dim=1)
            
            # coords l1 loss
            coord_loss = self.l1_loss(coords_, coords_w_noise)
            # cls kl loss
            num_classes = cls_.shape[-1]
            cls_flatten = torch.reshape(cls_, (-1, num_classes))
            cls_w_noise_flatten = torch.reshape(cls_w_noise, (-1, num_classes))
            cls_cons_loss = kl_div_adv(cls_flatten, cls_w_noise_flatten.detach())
            new_losses[f"cls_cons_loss"] = 10 * cls_cons_loss
            new_losses[f"box_cons_loss"] = 10 * coord_loss
                
            return new_losses

        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

        # DETR encoder and decoder forward
        if self.with_query_head:
            bbox_losses, x, outs, outs_w_noise = self.query_head.loss(x, x_noise, batch_data_samples)
            outputs_classes_w_noise, outputs_coords_with_noise = outs_w_noise
            outputs_classes, outputs_coords = outs
            cons_losses = consistent_loss(outputs_classes, outputs_classes_w_noise, outputs_coords, outputs_coords_with_noise)
            losses.update(bbox_losses)
            losses.update(cons_losses)

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

        img_feats = self.extract_feat(batch_inputs, batch_inputs2)
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
class EAEF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_pool = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim*2,dim*2,kernel_size=7,padding=3,groups=dim)
        self.ecse = Channel_Attention(dim*2)
        # self.ccse = Channel_Attention(dim)
        self.sse_r = Spatial_Attention(dim)
        self.sse_t = Spatial_Attention(dim)
    def forward(self, x):
        ############################################################################
        RGB,T = x[0], x[1]
        b, c, h, w = RGB.size()
        rgb_y = self.mlp_pool(RGB)
        t_y = self.mlp_pool(T)
        rgb_y = rgb_y / rgb_y.norm(dim=1, keepdim=True)
        t_y = t_y / t_y.norm(dim=1, keepdim=True)
        rgb_y = rgb_y.view(b, c, 1)
        t_y = t_y.view(b, 1, c)
        logits_per = c * rgb_y @ t_y
        cross_gate = torch.diagonal(torch.sigmoid(logits_per)).reshape(b, c, 1, 1)
        add_gate = torch.ones(cross_gate.shape).cuda() - cross_gate
        ##########################################################################
        New_RGB_e = RGB * cross_gate
        New_T_e = T * cross_gate
        New_RGB_c = RGB * add_gate
        New_T_c = T * add_gate
        x_cat_e = torch.cat((New_RGB_e, New_T_e), dim=1)
        ##########################################################################
        fuse_gate_e = torch.sigmoid(self.ecse(self.dwconv(x_cat_e)))
        rgb_gate_e, t_gate_e = fuse_gate_e[:, 0:c, :], fuse_gate_e[:, c:c * 2, :]
        ##########################################################################
        New_RGB = New_RGB_e * rgb_gate_e + New_RGB_c
        New_T = New_T_e * t_gate_e + New_T_c
        ##########################################################################
        New_fuse_RGB = self.sse_r(New_RGB)
        New_fuse_T = self.sse_t(New_T)
        attention_vector = torch.cat([New_fuse_RGB, New_fuse_T], dim=1)
        attention_vector = torch.softmax(attention_vector, dim=1)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        New_RGB = New_RGB * attention_vector_l
        New_T = New_T * attention_vector_r
        # New_fuse = New_T + New_RGB
        out = New_RGB, New_T
        ##########################################################################
        return out
class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2).view(b,c)
        return y
class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gap_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2)
        return max_out

class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1,bias=True)
    def forward(self, x):
        x1 = self.conv1(x)
        return x1
    