_base_ = ['co_dino_5scale_swin_l_16xb1_1x_coco_dual_more_data.py']

#find_unused_parameters=True
pretrained = 'swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = '/root/workspace/data/dual_mmdetection/mmdetection/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size)
]
# model settings
model = dict(
    type='CoDETR_Dual_Swin',
    data_preprocessor=dict(
        type='DoubleInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='Dual_SwinTransformer_C2Former',
        pretrain_img_size=384,
        embed_dims=192,
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
        init_cfg=None,),
       # init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 12
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


# LSJ + CopyPaste
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    # dict(type='Pre_Pianyi', canvas_size = (670, 542), p=1),
    dict(type='Pre_Pianyi', canvas_size = (670, 540), p=1),
    dict(type='BBox_Jitter', max_shift_px = 3, prob = 0.5),
    # dict(type='BBox_Jitter'),
    # dict(type='BBox_Jitter'),

    # dict(type='CopyPaste_Possion', img_scale=(640, 640)),
    dict(type='Image2Broadcaster',
        transforms=[
                dict(
                    type='RandomResize',
                    scale=image_size,
                    ratio_range=(0.1, 2.0),
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=image_size,
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(type='RandomFlip', prob=0.5),
        ]
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Branch',
         transforms=[
                 dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
        ]
    ),
    
]

train_pipeline = load_pipeline + [
    # dict(type='CopyPaste', max_num_pasted=100),
    
    dict(type='DoublePackDetInputs')
]

# train_dataloader = dict(
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#             pipeline=train_pipeline,
#             type=dataset_type,
#             metainfo=dict(classes=classes),
#             data_root=data_root,
#             ann_file='train.json',
#             data_prefix=dict(img='train/rgb'),
#         )
# )

train_dataloader = dict(
        batch_size=2, num_workers=1, 
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            # type=dataset_type,
            # metainfo=dict(classes=classes),
            # data_root=data_root,
            # ann_file='merged_coco_new.json',
            # data_prefix=dict(img='train_more/rgb'),
            pipeline=train_pipeline
        )
    )

# train_dataloader = dict(
#         batch_size=1
#     )