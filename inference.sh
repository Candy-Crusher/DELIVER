export PYTHONPATH="/home/xiaoshan/work/adap_v/DELIVER"
export CUDA_VISIBLE_DEVICES=1,3
eval_dataset='day'
input_type='rgb'
python tools/val_mm.py \
    --cfg configs/dsec_${input_type}_${eval_dataset}.yaml \
    --scene dsec_${input_type}_${eval_dataset} \
    --model_path output/DSEC_CMNeXt-B2_i/CMNeXt_CMNeXt-B2_DSEC_epoch135_68.51.pth \
    # --cfg configs/dsec_rgbe.yaml
    # --cfg configs/deliver_rgbdel.yaml
