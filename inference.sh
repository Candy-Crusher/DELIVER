export PYTHONPATH="/home/xiaoshan/work/adap_v/DELIVER"
export CUDA_VISIBLE_DEVICES=3
eval_dataset='day'
input_type='rgb'
python tools/val_mm.py \
    --cfg configs/dsec_${input_type}_${eval_dataset}.yaml \
    --scene dsec_${input_type}_${eval_dataset} \
    --classes 11 \
    --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch253_71.74.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch267_70.8.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch334_70.58.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch288_70.62.pth
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch356_73.55.pth \
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch105_62.93.pth
    # --model_path output/DSEC_CMNeXt-B2_ie/model_all_11_CMNeXt_CMNeXt-B2_DSEC_epoch204_73.29.pth \
    # --model_path output/DSEC_CMNeXt-B2_i/model_all_11_CMNeXt_CMNeXt-B2_DSEC_epoch196_73.44.pth \
    # --model_path output/DSEC_CMNeXt-B2_ie/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch360_73.32.pth \
    # --model_path output/DSEC_CMNeXt-B2_ie/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch245_72.59.pth \
    # --model_path output/DSEC_CMNeXt-B2_i/model_night_11_CMNeXt_CMNeXt-B2_DSEC_epoch340_68.88.pth \
    # --model_path output/DSEC_CMNeXt-B2_i/model_day_11_CMNeXt_CMNeXt-B2_DSEC_epoch327_72.71.pth \
    # --cfg configs/dsec_rgbe.yaml
    # --cfg configs/deliver_rgbdel.yaml
