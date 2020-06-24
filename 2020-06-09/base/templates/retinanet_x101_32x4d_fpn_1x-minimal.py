# Template for RetinaNet X101 FPN
# the base configuration to inherit from
_base_ = "/mmdetection/configs/retinanet/retinanet_x101_32x4d_fpn_1x_coco.py"
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
# this type uses the MMDET_CLASSES environment variable
dataset_type = 'Dataset'
# the root directory for train/test/val images and json files
data_root = '/data/buttercup_whole_only-2020-05-01'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/train.json',
        img_prefix=data_root + '/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/val.json',
        img_prefix=data_root + '/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/test.json',
        img_prefix=data_root + '/',
        pipeline=test_pipeline))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
# output a checkpoint every X epochs
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
# the number of epochs to train
total_epochs = 10
dist_params = dict(backend='nccl')
log_level = 'INFO'
# the directory to output the log files, checkpoints and fully expanded config file
work_dir = '/output/buttercup_whole_only-2020-05-01-retinanet_x101.test'
# the pre-trained model to use
load_from = '/models/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth'
resume_from = None
workflow = [('train', 1)]
