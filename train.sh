export PYTHONPATH="/home/xiaoshan/work/adap_v/DELIVER"
export CUDA_VISIBLE_DEVICES=0,2
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 \
  tools/train_mm.py \
  --cfg configs/dsec_rgb_night.yaml \
  --scene dsec_rgb_night \
  # --cfg configs/dsec_rgbe.yaml
  # --cfg configs/deliver_rgbdel.yaml

# NCCL_P2P_DISABLE=1
# export PYTHONPATH="/home/xiaoshan/work/adap_v/DELIVER"
# CUDA_VISIBLE_DEVICES=2,3
# python -m torch.distributed.launch --nproc_per_node=2 \
#     --use_env tools/train_mm.py \
#     --cfg configs/dsec_rgbe_day.yaml \
#     --scene dsec_rgbe_night \
#     # --cfg configs/dsec_rgbe.yaml
#     # --cfg configs/deliver_rgbdel.yaml
