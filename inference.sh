export PYTHONPATH="/home/xiaoshan/work/adap_v/DELIVER"
export CUDA_VISIBLE_DEVICES=0,2
eval_dataset='night'
input_type='rgb'
python tools/val_mm.py \
    --cfg configs/dsec_${input_type}_${eval_dataset}.yaml \
    --scene dsec_${input_type}_${eval_dataset} \
    --classes 12 \
    --model_path output/DSEC_CMNeXt-B2_i/model_night_12_CMNeXt_CMNeXt-B2_DSEC_epoch149_63.83.pth \
    # --cfg configs/dsec_rgbe.yaml
    # --cfg configs/deliver_rgbdel.yaml
