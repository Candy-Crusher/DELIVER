export PYTHONPATH="/home/xiaoshan/work/adap_v/DELIVER"
CUDA_VISIBLE_DEVICES=1,2,3
python tools/val_mm.py \
    --cfg configs/dsec_rgbe_night.yaml \
    --scene dsec_rgbe_night \
    # --cfg configs/dsec_rgbe.yaml
    # --cfg configs/deliver_rgbdel.yaml
