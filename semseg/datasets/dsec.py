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
        self.files = sorted(glob.glob(os.path.join(*[root, 'gtFine_next', split, '*', '*_gtFine_labelTrainIds11.png'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        print(f"Found {len(self.files)} {split} {case} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # 50%的概率inverse
        reverse_flag = False
        if random.random() < 0.5:
            lbl_path = str(self.files[index])
            event_path = get_new_name(lbl_path, idx_diff=-1).replace('/gtFine_next', '/startF1_img_event_50ms/event_20').replace('_gtFine_labelTrainIds11.png', '.npy')
            rgb = event_path.replace('/startF1_img_event_50ms/event_20', '/leftImg8bit').replace('.npy', '.png')
            flow = rgb.replace('/leftImg8bit', '/flow').replace('.png', '.npy')
            rgb_ref = lbl_path.replace('/gtFine_next', '/leftImg8bit_next').replace('_gtFine_labelTrainIds11.png', '.png')
        else:
            reverse_flag = True
            lbl_path_next = str(self.files[index])
            lbl_path = get_new_name(lbl_path_next, idx_diff=-1).replace('/gtFine_next', '/gtFine_cur')
            event_path = lbl_path.replace('/gtFine_cur', '/startF1_img_event_50ms/event_20').replace('_gtFine_labelTrainIds11.png', '.npy')
            rgb = lbl_path_next.replace('/gtFine_next', '/leftImg8bit_next').replace('_gtFine_labelTrainIds11.png', '.png')
            flow = rgb.replace('/leftImg8bit_next', '/flow_reverse').replace('.png', '.npy')
            rgb_ref = lbl_path.replace('/gtFine_cur', '/leftImg8bit').replace('_gtFine_labelTrainIds11.png', '.png')
        if self.n_classes == 12:
            lbl_path = lbl_path.replace('_gtFine_labelTrainIds11.png', '_gtFine_labelTrainIds12.png')
        elif self.n_classes == 19:
            lbl_path = lbl_path.replace('_gtFine_labelTrainIds11.png', '_gtFine_labelTrainIds.png')
        # lbl_path = lbl_path.split('.')[0]  # 获取文件名的基础部分（去掉扩展名）
        # lbl_path = f"{lbl_path}_gtFine_labelTrainIds11.png"  # 添加后缀并重新组合
        seq_name = Path(rgb).parts[-2]
        seq_idx = Path(rgb).parts[-1].split('_')[0]

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...][:, :440]
        H, W = sample['img'].shape[1:]
        sample['img_next'] = io.read_image(rgb_ref)[:3, ...][:, :440]
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        sample['mask'] = label[:, :440]
        event_voxel = np.load(event_path, allow_pickle=True)
        event_voxel = torch.from_numpy(event_voxel[:, :440])
        if reverse_flag:
            event_voxel = torch.flip(event_voxel, dims=[0])
        event_voxel = torch.cat([event_voxel[4*i:4*(i+1)].mean(0).unsqueeze(0) for i in range(5)], dim=0)
        sample['event'] = event_voxel
        flow = np.load(flow, allow_pickle=True)
        # print(flow.shape)   # 2 440 640
        # exit(0)
        sample['flow'] = torch.from_numpy(flow[:, :440])
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        img_next = sample['img_next']
        del sample['img_next']
        event_voxel = sample['event']
        del sample['event']
        flow = sample['flow']
        del sample['flow']

        sample = [sample[k] for k in self.modals]
        sample.append(event_voxel)
        sample.append(img_next)
        sample.append(flow)
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