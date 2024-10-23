import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
# import Image
from PIL import Image

def concatenate_images(images, direction='horizontal'):
    if not images:
        raise ValueError("The images list should not be empty")

    if direction == 'horizontal':
        # 水平拼接
        total_width = sum(image.width for image in images)
        max_height = max(image.height for image in images)
        new_image = Image.new('RGB', (total_width, max_height))

        current_x = 0
        for image in images:
            new_image.paste(image, (current_x, 0))
            current_x += image.width

    elif direction == 'vertical':
        # 垂直拼接
        max_width = max(image.width for image in images)
        total_height = sum(image.height for image in images)
        new_image = Image.new('RGB', (max_width, total_height))

        current_y = 0
        for image in images:
            new_image.paste(image, (0, current_y))
            current_y += image.height

    else:
        raise ValueError("Direction should be 'horizontal' or 'vertical'")

    return new_image

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2]*1)), int(ceil(image_size[3]*1)))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)

@torch.no_grad()
def evaluate(model, dataloader, device, save_dir=None, palette=None):
    print('Evaluating...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    for seq_names, seq_index, images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        event_voxel = images[1]
        rgb_next = images[2]
        flow = images[3]
        # label_ref = images[3]
        # psi = images[4]
        images = [images[0]]
        if sliding:
            preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
        else:
            # preds, _ , _ = model(images, event_voxel, rgb_next, flow, psi)
            preds, _ = model(images, event_voxel, rgb_next, flow)
            # preds = preds.softmax(dim=1)
            # preds = label_ref
            # print(preds.shape)
            # print
            # # H W转化为 19 H W
            # preds = F.one_hot(preds, n_classes).permute(0, 3, 1, 2).float()
            # print(preds.shape)


        # 保存图像
        if save_dir is not None:
            for i, idx in enumerate(seq_index):
                # 把shape为(19, H, W)的预测结果转换为(H, W)的numpy数组
                save_path = Path(save_dir) / seq_names[i]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                pred_argmax = preds[i].argmax(dim=0).cpu().numpy().astype(np.uint8)
                rgb_pred = palette[pred_argmax]
                rgb_lbl = palette[labels[i].cpu().numpy().astype(np.uint8)]
                # 将numpy数组转换为PIL图像
                pred_argmax = Image.fromarray(pred_argmax)
                rgb_pred = Image.fromarray(rgb_pred.astype(np.uint8))
                rgb_lbl = Image.fromarray(rgb_lbl.astype(np.uint8))
                img = Image.fromarray((images[0][i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pred_argmax.save(save_path / idx.replace('.npy', '_labelTrainIds11.png'))
                concatenated_image = concatenate_images([img,rgb_lbl,rgb_pred], direction='horizontal')
                concatenated_image.save(save_path / idx.replace('.npy', '_color.png'))
                # rgb_image.save(save_path / idx.replace('.npy', '_color.png'))
                # rgb_lbl.save(save_path / idx.replace('.npy', '_color_gt.png'))
        metrics.update(preds, labels)
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg, scene, classes, model_path):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None] # all
    
    model_path = Path(model_path)
    if not model_path.exists(): 
        print(model_path)
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_dir  = os.path.join(os.path.dirname(model_path), '{}_eval_{}'.format(scene, exp_time))
    eval_path = os.path.join(os.path.dirname(model_path), '{}_eval_{}.txt'.format(scene, exp_time))

    for case in cases:
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', classes, transform, cfg['DATASET']['MODALS'], case)
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case)

        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'])
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cuda'))
        print(msg)
        model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=eval_cfg['BATCH_SIZE'], pin_memory=False, sampler=sampler_val)
        if True:
            if eval_cfg['MSF']['ENABLE']:
                acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
            else:
                acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device, save_dir, palette=np.array(dataset.SEGMENTATION_CONFIGS[classes]["PALETTE"]))

            table = {
                'Class': list(dataset.SEGMENTATION_CONFIGS[classes]["CLASSES"]) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Acc': acc + [macc]
            }
            print("mIoU : {}".format(miou))
            print("Results saved in {}".format(model_path))

        with open(eval_path, 'a+') as f:
            f.writelines(str(model_path))
            f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    parser.add_argument('--scene', type=str, default='night')
    parser.add_argument('--model_path', type=str, default='night')
    parser.add_argument('--classes', type=int, default=11)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)
    main(cfg, args.scene, args.classes, args.model_path)
