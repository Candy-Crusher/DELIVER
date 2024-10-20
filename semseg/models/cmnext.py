import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.heads import LightHamHead
from semseg.models.heads import UPerHead
from semseg.models.modules.flow_network import unet
from semseg.models.modules.flow_network.FRMA.modified_frma import EventFlowEstimator
from semseg.models.modules.flow_network.FRMA.model import flow_network
from semseg.models.modules.flow_network.FRMA.config import Config
from semseg.models.modules.softsplat.frame_synthesis import *
from semseg.utils.pac import SupervisedGaussKernel2d
from semseg.losses import calc_photometric_loss, reduce_photometric_loss, LapLoss, VGGLoss
from fvcore.nn import flop_count_table, FlopCountAnalysis
import matplotlib.pyplot as plt

class DenseLayer(torch.nn.Module):
    def __init__(self, dim, growth_rate, bias):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(dim, growth_rate, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(torch.nn.Module):
    def __init__(self, dim, growth_rate, num_dense_layer, bias):
        super(RDB, self).__init__()
        self.layer = [DenseLayer(dim=dim+growth_rate*i, growth_rate=growth_rate, bias=bias) for i in range(num_dense_layer)]
        self.layer = torch.nn.Sequential(*self.layer)
        self.conv = nn.Conv3d(dim+growth_rate*num_dense_layer, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv(out)
        out = out + x

        return out

class RRDB(nn.Module):
    def __init__(self, dim, num_RDB, growth_rate, num_dense_layer, bias):
        super(RRDB, self).__init__()
        self.RDBs = nn.ModuleList([RDB(dim=dim, growth_rate=growth_rate, num_dense_layer=num_dense_layer, bias=bias) for _ in range(num_RDB)])
        # self.conv = nn.Sequential(*[nn.Conv3d(dim * num_RDB, dim, kernel_size=1, padding=0, stride=1, bias=bias),
        #                             nn.Conv3d(dim, dim, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=bias)])
        self.conv = nn.Conv3d(dim * num_RDB, dim, kernel_size=1, padding=0, stride=1, bias=bias)
        self.merge = nn.Conv2d(dim * 5, 5, kernel_size=3, padding=1, stride=1, bias=bias)
        self.shortcut = nn.Conv2d(dim * 5, 5, kernel_size=3, padding=1, stride=1, bias=bias)

    def forward(self, x):
        input = x
        RDBs_out = []
        for rdb_block in self.RDBs:
            x = rdb_block(x)
            RDBs_out.append(x)
        x = self.conv(torch.cat(RDBs_out, dim=1))
        x = self.merge(x.flatten(1, 2))
        input = self.shortcut(input.flatten(1, 2))
        return x + input
    
class CMNeXt(BaseModel):
    def __init__(self, backbone: str = 'CMNeXt-B0', num_classes: int = 25, modals: list = ['img', 'depth', 'event', 'lidar']) -> None:
        super().__init__(backbone, num_classes, modals, with_events=False)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        # self.flow_net = EventFlowEstimator(in_channels=4, num_multi_flow=1)
        # self.flow_net = unet.UNet(5, 2, False)
        # self.event_feature_extractor = nn.Sequential(nn.Conv3d(4, 4, kernel_size=[1,3,3], padding=[0,1,1], stride=1, bias=False),
        #                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                                 RRDB(dim=4, num_RDB=8, growth_rate=12, num_dense_layer=4, bias=False))
        feature_dims = [64, 128, 320, 512]
        self.softsplat_net = Synthesis(feature_dims, activation='PReLU')
        # self.flow_nets = nn.ModuleList(
        #     flow_network(config=Config('semseg/models/modules/flow_network/FRMA/experiment.cfg'), feature_dim=feature_dims[i])
        #     for i in range(len(feature_dims))
        # )
        # self.gauss_supervisor = SupervisedGaussKernel2d(kernel_size=3, stride=1, padding=1, dilation=1)
        self.apply(self._init_weights)

    # def forward(self, x: list, event_voxel: Tensor=None, rgb_next: Tensor=None, flow: Tensor=None, psi: Tensor=None) -> list:
    def forward(self, x: list, event_voxel: Tensor=None, rgb_next: Tensor=None, flow: Tensor=None) -> list:
        ## backbone
        # feature_before, event_feature_before = self.backbone(x, [event_voxel])
        feature_before = self.backbone(x)
        # feature_next = self.backbone([rgb_next])
        
        # feature_loss = 0
        # # flownet
        # # timelens unet + softsplat
        # event_voxel = event_voxel.unfold(1, 4, 4).permute(0, 4, 1, 2, 3)
        # event_voxel = self.event_feature_extractor(event_voxel)
        # b c t h w -> b ct h w

        # # flow = self.flow_net(event_voxel)
        # B, C, H ,W = x[0].shape
        # # flow = torch.zeros(B, 2, H, W).to(x[0].device)
        # # 可视化特征和光流
        # # 可视化feature在四个子图里

        # feature_after, interFlow = self.softsplat_net(feature_before, x[0], flow)
        feature_after, interFlow = self.softsplat_net(feature_before, x[0], event_voxel, flow)
        # feature_after = [self.gauss_supervisor(f, fn) for f, fn in zip(feature_after, feature_next)]
        # for i, fea in enumerate(feature_before):
        #     feature_after[i] = feature_after[i] + fea

        # feature_after = feature_before
        # feature_after, interFlow = self.softsplat_net(feature_before, flow, event_voxel, psi=psi)
        # # if residual
        # for i, fea in enumerate(feature_before):
        #     feature_after[i] = feature_after[i] + fea
        # # end

        # ## FRAM
        # # event_voxel B T H W
        # # T = 20
        # # 变成[[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]]这样 B C=4 T=5 H W的shape
        # feature_after = [None, None, None, None]
        # interFlow = [None, None, None, None]
        # event_voxel = event_voxel.unfold(1, 4, 4).permute(0, 4, 1, 2, 3)
        # for i, fea in enumerate(feature_before):
        #     feature_after[i], interFlow[i] = self.flow_nets[i](event_voxel, fea)

        # self.visualize_all([x[0]]+feature_before, [rgb_next]+feature_after, [rgb_next]+feature_next, [flow]+interFlow)
        # self.visualize_all([x[0]]+feature_before, feature_after, [rgb_next]+feature_next, interFlow)
        # exit(0)  
        ## 计算监督损失
        # loss_fn = nn.MSELoss()
        # loss_fn = LapLoss()
        # loss_fn = VGGLoss()
        # feature_loss = sum(loss_fn(f, fn) for f, fn in zip(feature_after, feature_next))
        # photometric_losses = [calc_photometric_loss(f, fn) for f, fn in zip(feature_after, feature_next)]
        # feature_loss = reduce_photometric_loss(photometric_losses)
        # feature_loss = loss_fn(feature_after, feature_next)
      ## decoder

        y = self.decode_head(feature_after)
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        # consistent_loss = 0
        # y_ref = self.decode_head(feature_next)
        # y_ref = F.interpolate(y_ref, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        # # L2 loss
        # consistent_loss = F.mse_loss(y, y_ref)
        # return y, feature_loss, consistent_loss
        return y
    
    def visualize_features(self, features, axes, title_prefix):
        for i, feature in enumerate(features):
            # 取第一个batch的特征图
            feature_map = feature[0].detach().cpu().numpy()
            # 取平均值以减少通道维度
            feature_map = feature_map.mean(0)
            h, w = feature_map.shape
            axes[i].imshow(feature_map, cmap='viridis', extent=[0, w, 0, h])
            axes[i].set_title(f'{title_prefix} Scale {i+1}')
            axes[i].axis('off')

    def visualize_flow(self, flows, axes, title_prefix):
        for i, flow in enumerate(flows):
            flow_map = flow[0].detach().cpu().numpy()
            flow_magnitude = (flow_map ** 2).sum(axis=0) ** 0.5
            axes[i].imshow(flow_magnitude, cmap='plasma')
            axes[i].set_title(f'{title_prefix} {i+1}', fontsize=10)
            axes[i].axis('off')

    def visualize_all(self, feature_before, feature_after, feature_next, interFlow):
        num_features = len(feature_before)
        fig, axes = plt.subplots(4, num_features, figsize=(20, 12))

        # 可视化特征图（处理前）
        self.visualize_features(feature_before, axes[0], 'Before')

        # 可视化特征图（处理后）
        self.visualize_features(feature_after, axes[1], 'After')

        # 可视化特征图（下一步）
        self.visualize_features(feature_next, axes[2], 'Next')

        # 可视化光流
        self.visualize_flow(interFlow, axes[3], 'Flow')

        plt.tight_layout(pad=2.0)
        plt.savefig('features_and_flow.png', dpi=150)
        plt.show()

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                # NOTE
                # msg = self.backbone.load_state_dict(checkpoint, strict=False)
                msg = self.load_state_dict(checkpoint, strict=False)
                # print("init_pretrained message: ", msg)

def load_dualpath_model(model, model_file):
    extra_pretrained = None
    if isinstance(extra_pretrained, str):
        raw_state_dict_ext = torch.load(extra_pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in raw_state_dict_ext.keys():
            raw_state_dict_ext = raw_state_dict_ext['state_dict']
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys(): 
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v

    if isinstance(extra_pretrained, str):
        for k, v in raw_state_dict_ext.items():
            if k.find('patch_embed1.proj') >= 0:
                state_dict[k.replace('patch_embed1.proj', 'extra_downsample_layers.0.proj.module')] = v 
            if k.find('patch_embed2.proj') >= 0:
                state_dict[k.replace('patch_embed2.proj', 'extra_downsample_layers.1.proj.module')] = v 
            if k.find('patch_embed3.proj') >= 0:
                state_dict[k.replace('patch_embed3.proj', 'extra_downsample_layers.2.proj.module')] = v 
            if k.find('patch_embed4.proj') >= 0:
                state_dict[k.replace('patch_embed4.proj', 'extra_downsample_layers.3.proj.module')] = v 
            
            if k.find('patch_embed1.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed1.norm', 'extra_downsample_layers.0.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed2.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed2.norm', 'extra_downsample_layers.1.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed3.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed3.norm', 'extra_downsample_layers.2.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed4.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed4.norm', 'extra_downsample_layers.3.norm.ln_{}'.format(i))] = v 
            elif k.find('block') >= 0:
                state_dict[k.replace('block', 'extra_block')] = v
            elif k.find('norm') >= 0:
                state_dict[k.replace('norm', 'extra_norm')] = v


    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict

if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    model = CMNeXt('CMNeXt-B2', 25, modals)
    model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    y = model(x)
    print(y.shape)
