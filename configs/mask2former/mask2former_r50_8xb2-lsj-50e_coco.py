_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco-panoptic.py']

num_things_classes = 24
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
# image_size = (1024, 1024)
image_size = (960, 540)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=False,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=False,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments)
model = dict(
    data_preprocessor=data_preprocessor,
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(1024,1024),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1980, 1080), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'

# train_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file='annotations/instances_val2017.json',
#         data_prefix=dict(img='val2017/'),
#         pipeline=test_pipeline))
# test_dataloader = val_dataloader

# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/instances_val2017.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = val_evaluator

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Cerebellum', 'Arachnoid', 'CN8', 'CN5', 'CN7','CN_9_10_11','SCA','AICA','SuperiorPetrosalVein',
           'Labrynthine','Vein','Brainstem','Suction','Bovie','Bipolar','Forcep','BluntProbe',
           'Drill','Kerrison','Cottonoid','Scissors','Unknown','Dissector','Teflon')
data_root='/home/nehal/code/mmdetection_supervised/mmdetection/data/mvd_surgery/'


backend_args = None

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

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

dataset_combined = dict(
    type=dataset_type,
    metainfo=dict(classes=classes),
    data_root='/home/nehal/Data/Neuro/zipfolders/CombinedCoco/',
    ann_file='annotations/combined_coco.json',
    data_prefix=dict(img='combined_images/'),
         pipeline=train_pipeline,
                    backend_args=backend_args)

# train_dataloader = dict(
#     batch_size=[1,1,1],
#     num_workers=2,
#     persistent_workers=True,    
#     sampler=dict(
#         type='MultiDataSampler',
#         dataset_ratio=[1,1,1]),
#     batch_sampler=dict(
#         type='MultiDataAspectRatioBatchSampler',
#         num_datasets=3),
#     dataset=dict(
#         type='ConcatDataset', datasets=[dataset_511B, dataset_511C, dataset_patient1a])) #, dataset_5171C, dataset_5172D,dataset_5172F]))


# train_dataloader = dict(
#     batch_size=[1,1,1,1,1,1],
#     num_workers=2,
#     persistent_workers=True,    
#     sampler=dict(
#         type='MultiDataSampler',
#         dataset_ratio=[1,1,1,1,1,1]),
#     batch_sampler=dict(
#         type='MultiDataAspectRatioBatchSampler',
#         num_datasets=6),
#     dataset=dict(
#         type='ConcatDataset', datasets=[dataset_511B,dataset_5172F, dataset_combined, dataset_patient1a, dataset_5171C, dataset_5172D])) #, dataset_5171C, dataset_5172D,dataset_5172F]))

train_dataloader = dict(
    batch_size=[1,1,1,1,1,1],
    num_workers=2,
    persistent_workers=True,    
    sampler=dict(
        type='MultiDataSampler',
        dataset_ratio=[1,1,1,1,1,1]),
    batch_sampler=dict(
        type='MultiDataAspectRatioBatchSampler',
        num_datasets=6),
    dataset=dict(
        type='ConcatDataset', datasets=[dataset_511B,dataset_5172F, dataset_combined, dataset_patient1a, dataset_5171C, dataset_5172D])) #, dataset_5171C, dataset_5172D,dataset_5172F]))

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         # explicitly add your class names to the field `metainfo`
#         metainfo=dict(classes=classes),
#         data_root='/home/nehal/Data/Neuro/zipfolders/CombinedCoco/',
#         ann_file='combined_coco.json',
#         data_prefix=dict(img='combined_images/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root='/home/nehal/Data/Neuro/zipfolders/CombinedCoco/',
        ann_file='combined_coco.json',
        data_prefix=dict(img='combined_images/',seg='annotations/'),
         pipeline=test_pipeline,
        backend_args=backend_args
        )
    )

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/nehal/Data/Neuro/zipfolders/CombinedCoco/' + 'annotations/combined_coco.json',
    metric= 'segm',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

