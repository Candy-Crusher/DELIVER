import os
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from semseg.metrics import Metrics
from val_mm import evaluate
import numpy as np
# import Image
from PIL import Image

def main(cfg, scene, classes, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 8
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    resume_flownet_path = cfg['MODEL']['RESUME_FLOWNET']
    gpus = int(os.environ['WORLD_SIZE'])

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', classes, traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', classes, valtransform, dataset_cfg['MODALS'])
    class_names = trainset.SEGMENTATION_CONFIGS[classes]["CLASSES"]

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'])
    resume_checkpoint = None
    if os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(resume_checkpoint, strict=False)
        # print("resume_checkpoint msg: ", msg)
        logger.info(msg)
    else:
        model.init_pretrained(model_cfg['PRETRAINED'])
    
    if os.path.isfile(resume_flownet_path):
        flow_net_type = model_cfg['FLOW_NET']
        if flow_net_type == 'eraft':
            ## for eraft
            flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))['model']
            if 'fnet.conv1.weight' in flownet_checkpoint:
                # delete weights of the first layer
                flownet_checkpoint.pop('fnet.conv1.weight')
                flownet_checkpoint.pop('fnet.conv1.bias')
            if 'cnet.conv1.weight' in flownet_checkpoint:
                # delete weights of the second layer
                flownet_checkpoint.pop('cnet.conv1.weight')
                flownet_checkpoint.pop('cnet.conv1.bias')
        elif flow_net_type == 'bflow':
            # for bflow
            flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))['state_dict']
            # 过滤掉 'flow_network.' 前缀
            # flownet_checkpoint = {k.replace('flow_network.', ''): v for k, v in flownet_checkpoint.items()}
            # 过滤掉 'net.' 前缀
            flownet_checkpoint = {k.replace('net.', ''): v for k, v in flownet_checkpoint.items()}
            if 'fnet_ev.conv1.weight' in flownet_checkpoint:
                # delete weights of the first layer
                flownet_checkpoint.pop('fnet_ev.conv1.weight')
                flownet_checkpoint.pop('fnet_ev.conv1.bias')
            # if 'cnet.conv1.weight' in flownet_checkpoint:
            #     # delete weights of the second layer
            #     flownet_checkpoint.pop('cnet.conv1.weight')
            #     flownet_checkpoint.pop('cnet.conv1.bias')

        msg = model.flow_net.load_state_dict(flownet_checkpoint, strict=False)
        # print("flownet_checkpoint msg: ", msg)
        logger.info(msg)

    model = model.to(device)
    
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs+1)*iters_per_epoch), sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None
    
    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        loss = resume_checkpoint['loss']        
        best_mIoU = resume_checkpoint['best_miou']
    
    # NOTE
    # # 冻结除 flow_net 和 softsplat_net 之外的所有层
    # for name, param in model.named_parameters():
    #     if not 'flow_net' in name and not 'softsplat_net' in name:
    #     # if not 'softsplat_net' in name:
    #         param.requires_grad = False
    # for name, param in model.named_parameters():
    #     if 'decode_head' in name:
    #         param.requires_grad = True
    # # 检查哪些参数被冻结了
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    # end

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=0, drop_last=True, pin_memory=True, sampler=sampler, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True, sampler=sampler_val, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=0, pin_memory=True, sampler=sampler_val, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model complexity =====================')
        # cal_flops(model, dataset_cfg['MODALS'], logger)
        logger.info('================== model structure =====================')
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)

    for epoch in range(start_epoch, epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0        
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        for iter, (seq_names, seq_index, sample, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)
            event_voxel = sample[1].to(device)
            # rgb_next = sample[2].to(device)
            # flow = sample[3].to(device)
            sample = [sample[0]]
            
            with autocast(enabled=train_cfg['AMP']):
                # logits = model(sample, event_voxel, rgb_next, flow)
                logits, feature_loss = model(sample, event_voxel)
                # logits, feature_loss = model(sample, event_voxel, rgb_next, flow)
                loss = loss_fn(logits, lbl) + 100*feature_loss
                # loss = loss_fn(logits, lbl) + 0.5*feature_loss + 0.5*consistent_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8 # minimum of lr
            train_loss += loss.item()
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        train_loss /= iter+1
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
        # torch.cuda.empty_cache()

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, _, _, ious, miou = evaluate(model, valloader, device)
                writer.add_scalar('val/mIoU', miou, epoch)

                if miou > best_mIoU:
                    prev_best_ckp = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_mIoU = miou
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deliver_rgbdel.yaml', help='Configuration file to use')
    parser.add_argument('--scene', type=str, default='night')
    parser.add_argument('--input_type', type=str, default='rgbe')
    parser.add_argument('--classes', type=int, default=11)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger(save_dir / f'{args.input_type}_{args.scene}_{args.classes}_{time_}_train.log')
    main(cfg, args.scene, args.classes, gpu, save_dir)
    cleanup_ddp()