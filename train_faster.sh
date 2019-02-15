config="configs/pascal_voc/ssd512_r50_voc.py"
./tools/dist_train.sh $config 4 --validate
#python ./tools/train.py $config
#./tools/dist_train.sh $config 4 #--resume_from work_dirs/faster_rcnn_r50_fpn_1x_voc0712/epoch_2.pth #--validate 
