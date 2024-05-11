_base_ = ['co_dino_5scale_r50_8xb2_1x_gaiic.py']


## Custom ##
# from .my_loading import LoadImageFromFile2
# from .my_wrapper import Image2Broadcaster, Branch
# from .my_formatting import DoublePackDetInputs
custom_imports = dict(imports=[
                                'projects.CO_DETR.codetr',
                                'mmdet.datasets.transforms.my_loading',
                                'mmdet.datasets.transforms.my_wrapper',
                                'mmdet.datasets.transforms.my_formatting',
                                'mmdet.models.data_preprocessors.my_data_preprocessor',
                                'mmdet.datasets.my_coco',
                                'projects.CO_DETR.codetr.codetr_dual_stream_reg_v2'
                               ], allow_failed_imports=False)

dataset_type = 'DualStreamCocoDataset'
data_root = '/root/workspace/data/GAIIC2024/'

pretrained = 'swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa

image_size = (1024, 1024)
num_classes = 5
classes = ('car', 'truck', 'bus', 'van', 'freight_car')

batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size)
]
# model settings
model = dict(
    # Dual Stream model
    type='CoDETR_Dual_Reg_V2',

    data_preprocessor=dict(
        type='DoubleInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192, # Dual-Stream features combined before neck
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),

    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)), 
        num_classes=num_classes,
        transformer=dict(encoder=dict(with_cp=6))),
    
    )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations'),
    
    dict(type='Image2Broadcaster',
        transforms=[
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                            (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                            (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                            (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                            (992, 2048), (1024, 2048), (1056, 2048),
                            (1088, 2048), (1120, 2048), (1152, 2048),
                            (1184, 2048), (1216, 2048), (1248, 2048),
                            (1280, 2048), (1312, 2048), (1344, 2048),
                            (1376, 2048), (1408, 2048), (1440, 2048),
                            (1472, 2048), (1504, 2048), (1536, 2048)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=image_size,
                    recompute_bbox=True,
                    allow_negative_crop=True),
                ]
         
         ),
    dict(type='Branch',
         transforms=[
                 dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
        ]
    ),
    dict(type='DoublePackDetInputs')
]

train_dataloader = dict(
        batch_size=2, num_workers=10, 
        dataset=dict(
            type=dataset_type,
            metainfo=dict(classes=classes),
            data_root=data_root,
            ann_file='train.json',
            data_prefix=dict(img='train/rgb'),
            pipeline=train_pipeline
            )
        )

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations'),
    
    dict(type='Branch',
         transforms=[
            dict(type='Resize', scale=image_size, keep_ratio=True),
         ]
    ),
    dict(type='Branch',
         transforms=[
            dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
        ]
    ),
    
    dict(
        type='DoublePackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'img_path2', 'ori_shape2', 'img_shape2',
                   'scale_factor'))
]

val_evaluator = dict(
    ann_file=data_root + 'val.json')

val_dataloader = dict(dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/rgb'),
        pipeline=test_pipeline))

test_dataloader = dict(dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='instances_test2017.json',
        data_prefix=dict(img='test/rgb'),
        pipeline=test_pipeline))

test_evaluator = dict(
        format_only=True,
        ann_file=data_root + 'instances_test2017.json',
        outfile_prefix='work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_reg_v2/test')

# optim_wrapper = dict(optimizer=dict(lr=1e-4))
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone1': dict(lr_mult=0.1), 'backbone2': dict(lr_mult=0.1)}))

max_epochs = 16
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
