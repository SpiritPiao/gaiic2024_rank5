_base_ = ['co_dino_5scale_r50_lsj_8xb2_1x_gaiic_dual_stream_more_data_albu_with_Vis_5cls_dark_enhance.py']


pretrained = 'swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = '/root/workspace/data/dual_mmdetection/mmdetection/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa
load_from = 'work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_albu/pre_521.pth'
# load_from = 'work_dirs/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_dual_swin_albu_with_Vis_3cls_drak_enhance/0519_dark_529.pth'
# model settings
model = dict(
    type='CoDETR_Dual_Swin',
    backbone=dict(
        _delete_=True,
        type='Dual_SwinTransformer',
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 7
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[5],
        gamma=0.1)
]

