# Template for Faster R-CNN ResNet101 FPN
# the base configuration the configuration is based on
_base_ = "/mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py"
# this type uses the MMDET_CLASSES environment variable
dataset_type = 'Dataset'
# the root directory for train/test/val images and json files
data_root = '/data/somewhere'
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
# the number of epochs to train
total_epochs = 10
# output a checkpoint every X epochs
checkpoint_config = dict(interval=5)
# the pre-trained model to use
load_from = '/models/faster_rcnn_r101_fpn_1x_20181129-d1468807.pth'
workflow = [('train', 1)]
# the directory to output the log files, checkpoints and fully expanded config file
work_dir = '/output/buttercup_whole_only-2020-05-01-faster_rcnn_fpn.test'
