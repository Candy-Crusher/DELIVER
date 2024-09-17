export PYTHONPATH="/home/xiaoshan/work/adap_v/DELIVER"
CUDA_VISIBLE_DEVICES=1
python tools/val_mm.py \
    --cfg configs/dsec_rgbe_day.yaml \
    --scene dsec_rgbe_day \
    # --cfg configs/dsec_rgbe.yaml
    # --cfg configs/deliver_rgbdel.yaml
