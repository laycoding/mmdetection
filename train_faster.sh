# CUDA_VISIBLE_DEVICES=5,6
#export PATH="/workspace/mnt/group/face/zhangweidong/ENV/anaconda3/bin:$PATH"
#source activate mmdet

config="configs/pascal_voc/refinedet512_r50_1x_voc.py"
./tools/dist_train.sh $config 4 --validate
#CUDA_VISIBLE_DEVICES=0 python ./tools/train.py $config

pkl_result="r50_ssd_result.pkl"
rm $pkl_result
checkpoint="work_dirs/ssd512_r50_voc/latest.pth"
python tools/test.py  $config $checkpoint --out $pkl_result
python tools/voc_eval.py $pkl_result $config


