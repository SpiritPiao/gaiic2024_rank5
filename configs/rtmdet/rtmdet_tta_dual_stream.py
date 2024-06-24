custom_imports = dict(imports=[
                               'mmdet.datasets.transforms.my_loading',
                               'mmdet.datasets.transforms.my_wrapper',
                               'mmdet.datasets.transforms.my_formatting',
                               'mmdet.models.data_preprocessors.my_data_preprocessor',
                               'mmdet.datasets.my_coco',
                               'mmdet.models.detectors.rtmdet_dual_stream',
                               'mmdet.datasets.transforms.my_transforms',
                               ], allow_failed_imports=False)

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

dict(
        type='Branch',
        transforms=[
            dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        ]
    ),

img_scales = [(640, 640), (320, 320), (960, 960)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadImageFromFile2'),

    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='Branch',
                    transforms=[dict(type='Resize', scale=s, keep_ratio=True)]
                )
                for s in img_scales
            ],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(
                    type='Branch',
                    transforms=[dict(type='RandomFlip', prob=1.)]
                ),
                dict(
                    type='Branch',
                    transforms=[dict(type='RandomFlip', prob=0.)]
                )
                
                # dict(type='RandomFlip', prob=1.),
                # dict(type='RandomFlip', prob=0.)
            ],
            [
                
                dict(
                    type='Branch',
                    transforms=[
                        dict(
                            type='Pad',
                            size=(960, 960),
                            pad_val=dict(img=(114, 114, 114))
                        )
                    ]
                ),
                
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='DoublePackDetInputs',
                    meta_keys=('img_id', 'img_path', 'img_path2','ori_shape', 'img_shape', 'ori_shape2', 'img_shape2',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]
