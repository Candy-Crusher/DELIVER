import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation
import re
import random

from scipy.ndimage import gaussian_filter

def backwarp(tenIn, tenFlow):
    tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[2], dtype=tenFlow.dtype).view(1, 1, -1).repeat(1, tenFlow.shape[1], 1)
    tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[1], dtype=tenFlow.dtype).view(1, -1, 1).repeat(1, 1, tenFlow.shape[2])
    tenGrid = torch.cat([tenHor, tenVer], 0)

    tenFlow = torch.cat([tenFlow[0:1, :, :] / ((tenIn.shape[2] - 1.0) / 2.0), tenFlow[1:2, :, :] / ((tenIn.shape[1] - 1.0) / 2.0)] , 0)

    return torch.nn.functional.grid_sample(input=tenIn.unsqueeze(0), grid=(tenGrid + tenFlow).permute(1,2,0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0)
# end

def compute_photometric_consistency(I0, I1, F0to1):
    """计算光度一致性 ψ_photo"""
    warped_I1 = backwarp(I1, F0to1)
    diff = I0 - warped_I1
    psi_photo = torch.sqrt(diff[0]**2+diff[1]**2+diff[2]**2)
    return psi_photo

def compute_flow_consistency(F0to1, F1to0):
    """计算光流一致性 ψ_flow"""
    # 反向映射光流 F1to0
    warped_F1to0 = backwarp(F1to0, F0to1)
    # 计算一致性
    diff = F0to1 - warped_F1to0
    psi_flow = torch.sqrt(diff[0]**2+diff[1]**2)
    return psi_flow

def compute_flow_variance(F0to1):
    """计算光流方差 ψ_varia"""
    F_squared = F0to1 ** 2
    G_F_squared = gaussian_filter(F_squared, sigma=1)
    
    G_F = gaussian_filter(F0to1, sigma=1)
    
    variance = torch.from_numpy(G_F_squared - (G_F ** 2))
    
    psi_varia = torch.sqrt(variance[0]+variance[1])
    return psi_varia

def get_new_name(filepath, idx_diff):
    # 正则表达式匹配文件名中的编号部分
    filename = os.path.basename(filepath)
    pattern = re.compile(r'(\d+)_gtFine_labelTrainIds11\.png')

    match = pattern.match(filename)
    if match:
        # 提取编号部分
        number = match.group(1)
        # 构建新的文件名
        new_filename = f'{int(number)+idx_diff:06d}_gtFine_labelTrainIds11.png'
        # 构建完整的路径名
        new_filepath = filepath.replace(f'{number}_gtFine_labelTrainIds11.png', new_filename)
        return new_filepath
    return None

class DSEC(Dataset):
    # 定义类别和调色板的字典
    SEGMENTATION_CONFIGS = {
        11: {
            "CLASSES": [
                "background", "building", "fence", "person", "pole",
                "road", "sidewalk", "vegetation", "car", "wall",
                "traffic sign",
            ],
            "PALETTE": torch.tensor([
                [0, 0, 0], [70, 70, 70], [190, 153, 153], [220, 20, 60], [153, 153, 153], 
                [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142], [102, 102, 156], 
                [220, 220, 0],
            ])
        },
        12: {
            "CLASSES": [
                "background", "building", "fence", "person", "pole",
                "road", "sidewalk", "vegetation", "car", "wall",
                "traffic sign", "curb",
            ],
            "PALETTE": torch.tensor([
                [0, 0, 0], [70, 70, 70], [190, 153, 153], [220, 20, 60], [153, 153, 153], 
                [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142], [102, 102, 156], 
                [220, 220, 0], [255, 170, 255],
            ])
        },
        19: {
            "CLASSES": [
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
                'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
            ],
            "PALETTE": torch.tensor([
                [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
            ]),
            "ID2TRAINID": {
                0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 
                20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, 34: 2, 35: 4, 36: 255, 37: 5, 38: 255, 39: 255, 
                40: 255, 41: 255, 42: 255, 43: 255, 44: 255, -1: 255
            }
        }
    }

    def __init__(self, root: str = 'data/DSEC', split: str = 'train', n_classes: int = 11, transform = None, modals = ['img', 'event'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = n_classes
        self.ignore_label = 255
        self.modals = modals
        # self.files = sorted(glob.glob(os.path.join(*[root, 'leftImg8bit', split, '*', '*.png'])))
        # self.files = sorted(glob.glob(os.path.join(*[root, 'gtFine_next', split, '*', '*_gtFine_labelTrainIds11.png'])))
        self.files = sorted(glob.glob(os.path.join(*[root, 'sample', split, '*', '*.npy'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        print(f"Found {len(self.files)} {split} {case} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # # 50%的概率inverse
        # # reverse_flag = False
        # # if random.random() < 0.5:
        # #     lbl_path = str(self.files[index])
        # #     event_path = get_new_name(lbl_path, idx_diff=-1).replace('/gtFine_next', '/startF1_img_event_50ms/event_20').replace('_gtFine_labelTrainIds11.png', '.npy')
        # #     rgb = event_path.replace('/startF1_img_event_50ms/event_20', '/leftImg8bit').replace('.npy', '.png')
        # #     flow = rgb.replace('/leftImg8bit', '/flow').replace('.png', '.npy')
        # #     rgb_ref = lbl_path.replace('/gtFine_next', '/leftImg8bit_next').replace('_gtFine_labelTrainIds11.png', '.png')
        # # else:
        # #     reverse_flag = True
        # #     lbl_path_next = str(self.files[index])
        # #     lbl_path = get_new_name(lbl_path_next, idx_diff=-1).replace('/gtFine_next', '/gtFine_cur')
        # #     event_path = lbl_path.replace('/gtFine_cur', '/startF1_img_event_50ms/event_20').replace('_gtFine_labelTrainIds11.png', '.npy')
        # #     rgb = lbl_path_next.replace('/gtFine_next', '/leftImg8bit_next').replace('_gtFine_labelTrainIds11.png', '.png')
        # #     flow = rgb.replace('/leftImg8bit_next', '/flow_reverse').replace('.png', '.npy')
        # #     rgb_ref = lbl_path.replace('/gtFine_cur', '/leftImg8bit').replace('_gtFine_labelTrainIds11.png', '.png')
        # lbl_path = str(self.files[index])
        # lbl_path_ref = get_new_name(lbl_path, idx_diff=-1).replace('/gtFine_next', '/gtFine_cur')
        # event_path = get_new_name(lbl_path, idx_diff=-1).replace('/gtFine_next', '/startF1_img_event_50ms/event_20').replace('_gtFine_labelTrainIds11.png', '.npy')
        # rgb = event_path.replace('/startF1_img_event_50ms/event_20', '/leftImg8bit').replace('.npy', '.png')
        # flow = rgb.replace('/leftImg8bit', '/flow').replace('.png', '.npy')
        # rgb_ref = lbl_path.replace('/gtFine_next', '/leftImg8bit_next').replace('_gtFine_labelTrainIds11.png', '.png')
        # flow_inverse = rgb_ref.replace('/leftImg8bit_next', '/flow_reverse').replace('.png', '.npy')

        # if self.n_classes == 12:
        #     lbl_path = lbl_path.replace('_gtFine_labelTrainIds11.png', '_gtFine_labelTrainIds12.png')
        # elif self.n_classes == 19:
        #     lbl_path = lbl_path.replace('_gtFine_labelTrainIds11.png', '_gtFine_labelTrainIds.png')
        # # lbl_path = lbl_path.split('.')[0]  # 获取文件名的基础部分（去掉扩展名）
        # # lbl_path = f"{lbl_path}_gtFine_labelTrainIds11.png"  # 添加后缀并重新组合
        # seq_name = Path(rgb).parts[-2]
        # seq_idx = Path(rgb).parts[-1].split('_')[0]

        # sample = {}
        # sample['img'] = io.read_image(rgb)[:3, ...][:, :440]
        # # H, W = sample['img'].shape[1:]
        # sample['img_next'] = io.read_image(rgb_ref)[:3, ...][:, :440]
        # label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        # label_ref = io.read_image(lbl_path_ref)[0,...].unsqueeze(0)
        # sample['mask'] = label[:, :440]
        # sample['mask_cur'] = label_ref[:, :440]
        # event_voxel = np.load(event_path, allow_pickle=True)
        # event_voxel = torch.from_numpy(event_voxel[:, :440])
        # sample['event'] = event_voxel
        # flow = np.load(flow, allow_pickle=True)
        # flow_inverse = np.load(flow_inverse, allow_pickle=True)
        # # # print(flow.shape)   # 2 440 640
        # # # exit(0)
        # sample['flow'] = torch.from_numpy(flow[:, :440])
        # sample['flow_inverse'] = torch.from_numpy(flow_inverse[:, :440])

        # # save dict
        # np.save(event_path.replace('/startF1_img_event_50ms/event_20', '/sample'), sample)
        sample_path = str(self.files[index])
        sample = np.load(sample_path, allow_pickle=True).item()
        # dict_keys(['img', 'img_next', 'mask', 'mask_cur', 'event', 'flow', 'flow_inverse'])
        seq_name = Path(sample_path).parts[-2]
        seq_idx = Path(sample_path).parts[-1].split('_')[0]
        # bin = 5
        # sample['event'] = torch.cat([sample['event'][bin*i:bin*(i+1)].mean(0).unsqueeze(0) for i in range(20//bin)], dim=0)

        if self.transform:
            sample = self.transform(sample)
        if random.random() < 0:
            label = sample['mask_cur']
            del sample['mask_cur']
            # flow zero
            flow = torch.zeros_like(sample['flow'])
            del sample['flow']
        else:
            label = sample['mask']
            del sample['mask']
            flow = sample['flow']
            del sample['flow']
        label = self.encode(label.squeeze().numpy()).long()
        # label_ref = sample['mask_cur']
        # del sample['mask_cur']
        # label_ref = self.encode(label_ref.squeeze().numpy()).long()
        event_voxel = sample['event']
        del sample['event']
        img_next = sample['img_next']
        del sample['img_next']

        # flow_inverse = sample['flow_inverse']
        # del sample['flow_inverse']

        # # 计算各个度量
        # psi_photo = compute_photometric_consistency(sample['img'], img_next, flow)
        # psi_flow = compute_flow_consistency(flow, flow_inverse)
        # psi_varia = compute_flow_variance(flow)
        # # 把这三个向量存成一个npy
        # psi = torch.stack([psi_photo, psi_flow, psi_varia], dim=0)
        # # np.save(event_path.replace('/startF1_img_event_50ms/event_20', '/psi'), psi)

        sample = [sample[k] for k in self.modals]
        sample.append(event_voxel)
        sample.append(img_next)
        sample.append(flow)
        # sample.append(label_ref)
        # sample.append(psi)
        return seq_name, seq_idx, sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    for case in cases:

        trainset = DELIVER(transform=traintransform, split='val', case=case)
        trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

        for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))