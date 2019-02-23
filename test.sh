pkl_result="r50_ssd_result.pkl"
rm $pkl_result
config="configs/pascal_voc/ssd512_r50_voc.py"
checkpoint="work_dirs/ssd512_r50_voc_t1/epoch_240.pth"
python tools/test.py  $config $checkpoint --out $pkl_result
python tools/voc_eval.py $pkl_result $config