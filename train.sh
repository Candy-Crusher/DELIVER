export PYTHONPATH="/home/xiaoshan/work/adap_v/DELIVER"
CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 \
    --use_env tools/train_mm.py \
    --cfg configs/dsec_rgbe.yaml
    # --cfg configs/dsec_rgbe.yaml
    # --cfg configs/deliver_rgbdel.yaml
