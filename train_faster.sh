CUDA_VISIBLE_DEVICES=0

# ./tools/dist_train.sh configs/faster_rcnn_r50_fpn_1x_voc.py 2 -validate 
python ./tools/train.py configs/faster_rcnn_r50_fpn_1x_voc.py
