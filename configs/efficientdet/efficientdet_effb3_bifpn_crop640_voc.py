_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa

model = dict(
    type='EfficientDet',
    arch='b0',
    backbone=dict(
        type='EfficientNet',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        # norm_cfg=dict(
        #     type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        # init_cfg=dict(
        #     type='Pretrained', prefix='backbone', checkpoint=checkpoint)
    ),
    neck=dict(
        type='BiFPN',
        in_channels=[40, 112, 320],
        norm_cfg=norm_cfg,
        upsample_cfg=dict(mode='nearest')),
    bbox_head=dict(
        type='EfficientHead',
        num_classes=2,
        num_ins=5,
        stacked_convs=4,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

img_size = 640
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
         img_scale=(img_size, img_size),
         multiscale_mode='range',
         ratio_range=(0.1, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=(img_size, img_size), allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(img_size, img_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(img_size, img_size)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.18, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    min_lr=1e-5)

runner = dict(type='EpochBasedRunner', max_epochs=300)
