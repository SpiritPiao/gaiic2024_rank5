_base_ = ['co_dino_5scale_r50_lsj_8xb2_1x_gaiic_dual_stream.py']

pretrained = 'swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'co_dino_5scale_swin_large_1x_coco-27c13da4.pth'
# load_from = 'work_dirs/co_dino_5scale_swin_l_16xb1_1x_coco_dual_Visdrone/epoch_12.pth'
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
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
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(transformer=dict(encoder=dict(with_cp=6))))

train_dataloader = dict(batch_size=2, num_workers=1)
