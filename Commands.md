## Training 
CUDA_VISIBLE_DEVICES=0,1,2 bash ./tools/dist_train.sh configs/solov2/solov2_x101-dcn_fpn_ms-3x_coco_surgery.py 3

CUDA_VISIBLE_DEVICES=0,1,2 bash ./tools/dist_train.sh configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery.py 3

CUDA_VISIBLE_DEVICES=0,1,2 bash ./tools/dist_train.sh /home/nehal/code/mmdetection_supervised/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery_new.py 3

## EVALUATION OR TESTING 

CUDA_VISIBLE_DEVICES=0  python tools/test.py /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/solov2_r50_fpn_1x_coco_surgery.py /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/epoch_9.pth

CUDA_VISIBLE_DEVICES=0,1,2 ./tools/dist_test.sh /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/solov2_r50_fpn_1x_coco_surgery.py /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/epoch_9.pth 3

CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/solov2_r50_fpn_1x_coco_surgery.py /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/epoch_9.pth 1 --work-dir TEST --out results.pkl --show-dir VIZ


CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh /home/nehal/code/mmdetection_supervised/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery.py /home/nehal/code/mmdetection_supervised/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery/best_coco_segm_mAP_iter_31000.pth 1 --work-dir TEST_mask2former_vid517_2F --out TEST_mask2former_vid517/results_mask2former_vid517_2F.pkl --show-dir VIZ

# test file has option to dump predictions as pickle file 

/home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11/frame_476.jpg

python demo/image_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11 /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/solov2_r50_fpn_1x_coco_surgery.py --weights /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/epoch_9.pth 

rsync -r --progress /home/nehal/code/mmdetection/outputs/vis nehaldilip@hnt-0649.vpn.private.upenn.edu:/Users/nehaldilip/Desktop


CUDA_VISIBLE_DEVICES=0 python demo/image_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11 /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/epoch_9.pth --pred-score-thr 0.75

CUDA_VISIBLE_DEVICES=0  python tools/test.py /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/solov2_r50_fpn_1x_coco_surgery.py /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/epoch_9.pth --cfg-options test_evaluator.ann_file=data/coco/annotations/videob.json

CUDA_VISIBLE_DEVICES=0 python demo/video_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11.mp4  /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/solov2_r50_fpn_1x_coco_surgery.py /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/epoch_9.pth --out VID002B_0.0.0_0.4.11result.mp4

 
 CUDA_VISIBLE_DEVICES=0 python demo/image_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11 /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/solov2_r50_fpn_1x_coco_surgery.py --weights /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery/epoch_9.pth --pred-score-thr 0.75

CUDA_VISIBLE_DEVICES=0 python demo/image_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11 /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery_debug/solov2_r50_fpn_1x_coco_surgery_debug.py --weights /home/nehal/code/mmdetection/work_dirs/solov2_r50_fpn_1x_coco_surgery_debug/epoch_6.pth --pred-score-thr 0.15

 CUDA_VISIBLE_DEVICES=3 python demo/image_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11 /home/nehal/code/mmdetection/work_dirs/solov2_x101-dcn_fpn_ms-3x_coco_surgery/solov2_x101-dcn_fpn_ms-3x_coco_surgery.py --weights /home/nehal/code/mmdetection/work_dirs/solov2_x101-dcn_fpn_ms-3x_coco_surgery/epoch_36.pth --pred-score-thr 0.75
 

 CUDA_VISIBLE_DEVICES=3 python demo/image_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11 /home/nehal/code/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery.py --weights /home/nehal/code/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery/best_coco_segm_mAP_iter_45000.pth --pred-score-thr 0.70
 

  CUDA_VISIBLE_DEVICES=3 python demo/image_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11 /home/nehal/code/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery.py --weights /home/nehal/code/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery/best_coco_segm_mAP_iter_75000.pth --pred-score-thr 0.70
 

  CUDA_VISIBLE_DEVICES=3 python demo/image_demo.py /home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11 /home/nehal/code/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery.py --weights /home/nehal/code/mmdetection/work_dirs/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_surgery/bestmodels/best_coco_segm_mAP_iter_45000.pth --pred-score-thr 0.70
 


# Command to browse dataset 
python tools/analysis_tools/browse_dataset.py ${CONFIG} [-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]