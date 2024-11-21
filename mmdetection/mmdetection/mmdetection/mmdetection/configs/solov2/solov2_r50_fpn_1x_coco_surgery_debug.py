# _base_ = [
#     '../_base_/datasets/coco_instance.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]
_base_ = [
    # '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]




# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Cerebellum', 'Arachnoid', 'CN8', 'CN5', 'CN7','CN_9_10_11','SCA','AICA','SuperiorPetrosalVein',
           'Labrynthine','Vein','Brainstem','Suction','Bovie','Bipolar','Forcep','BluntProbe',
           'Drill','Kerrison','Cottonoid','Scissors','Unknown')
data_root='/home/nehal/code/mmdetection/data/mvd_surgery/'


backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_511B=dict(
    type=dataset_type,
    metainfo=dict(classes=classes),
    data_root=data_root,
    ann_file='annotations/vid511B_coco.json',
    data_prefix=dict(img='images_511B/'),
        filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
                    backend_args=backend_args)

dataset_511C =dict(
    type=dataset_type,
        metainfo=dict(classes=classes),
                data_root=data_root,
    ann_file='annotations/vid511C_coco.json',
    data_prefix=dict(img='images_511C/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
                    backend_args=backend_args)
    
dataset_patient1a =dict(
    type=dataset_type,
            metainfo=dict(classes=classes),
    data_root=data_root,
    ann_file='annotations/vidPatient1_coco.json',
    data_prefix=dict(img='images_patient1a/'),
     pipeline=train_pipeline,
                    backend_args=backend_args)
    



dataset_5171C =dict(
    type=dataset_type,
        metainfo=dict(classes=classes),
    data_root=data_root,
    ann_file='annotations/vid517_1C_coco.json',
    data_prefix=dict(img='images_VID517_1C/'),
         pipeline=train_pipeline,
                    backend_args=backend_args)
dataset_5172D =dict(
    type=dataset_type,
        metainfo=dict(classes=classes),

    data_root=data_root,
    ann_file='annotations/vid517_2D_coco.json',
    data_prefix=dict(img='images_VID517_2D/'),
         pipeline=train_pipeline,
                    backend_args=backend_args)
dataset_5172F = dict(
    type=dataset_type,
        metainfo=dict(classes=classes),
    data_root=data_root,
    ann_file='annotations/vid517_2F_coco.json',
    data_prefix=dict(img='images_VID517_2F/'),
         pipeline=train_pipeline,
                    backend_args=backend_args)


train_dataloader = dict(
    batch_size=[1,1],
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='MultiDataSampler',
        dataset_ratio=[1,1]),
    batch_sampler=dict(
        type='MultiDataAspectRatioBatchSampler',
        num_datasets=2),
    dataset=dict(
        type='ConcatDataset', datasets=[dataset_511B, dataset_511C])) #, dataset_5171C, dataset_5172D,dataset_5172F]))


                                  

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         # explicitly add your class names to the field `metainfo`
#         metainfo=dict(classes=classes),
#         data_root=data_root,
#         ann_file='annotations/vidPatient1_coco.json',
#         data_prefix=dict(img='images_patient1a/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args=backend_args))

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         # explicitly add your class names to the field `metainfo`
#         metainfo=dict(classes=classes),
#         data_root=data_root,
#         ann_file=data_root+'annotations/vid517_2F_coco.json',
#         data_prefix=dict(img=data_root+'images_VID517_2F')
#         )
# )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/vid511B_coco.json',
        data_prefix=dict(img='images_511B/'),
         pipeline=test_pipeline,
        backend_args=backend_args
        )
    )

test_dataloader = val_dataloader




# model settings
model = dict(
    type='SOLOv2',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=22,
        in_channels=256,
        feat_channels=512,
        stacked_convs=4,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=30))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.005), clip_grad=dict(max_norm=35, norm_type=2))

# val_evaluator = dict(metric='segm',ann_file=data_root + 'annotations/vidPatient1_coco.json')
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/vid511B_coco.json',
    metric= 'segm',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=3)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=3),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=True,interval=5,show=False))