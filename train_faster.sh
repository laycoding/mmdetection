# CUDA_VISIBLE_DEVICES=5,6
#export PATH="/workspace/mnt/group/face/zhangweidong/ENV/anaconda3/bin:$PATH"
#source activate mmdet

config="configs/pascal_voc/refinedet512_r50_1x_voc.py"
#./tools/dist_train.sh $config 4 --validate
CUDA_VISIBLE_DEVICES=5 python ./tools/train.py $config
