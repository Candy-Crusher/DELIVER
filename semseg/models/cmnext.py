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
from semseg.models.modules.flow_network.eraft.eraft import ERAFT
from semseg.models.modules.flow_network.bflow.raft_spline.raft import RAFTSpline
from semseg.models.modules.softsplat.frame_synthesis import *
from semseg.models.modules.softsplat.softsplat import *
from semseg.models.modules.memory.memory_encoder import *
from semseg.utils.pac import SupervisedGaussKernel2d
from semseg.losses import calc_photometric_loss, reduce_photometric_loss, LapLoss, VGGLoss, outlier_penalty_loss
from fvcore.nn import flop_count_table, FlopCountAnalysis
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import moviepy.editor
    
class CMNeXt(BaseModel):
    def __init__(self, backbone: str = 'CMNeXt-B0', num_classes: int = 25, modals: list = ['img', 'depth', 'event', 'lidar'], backbone_flag: bool=False, flow_net_flag: bool=False) -> None:
        super().__init__(backbone, num_classes, modals, with_events=False,backbone_flag=backbone_flag,  flow_net_flag=flow_net_flag)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        if self.flow_net_flag:
            # self.flow_net = flow_network(config=Config('semseg/models/modules/flow_network/FRMA/experiment.cfg'), feature_dim=3)
            # # self.flow_nets = nn.ModuleList(
            #     flow_network(config=Config('semseg/models/modules/flow_network/FRMA/experiment.cfg'), feature_dim=feature_dims[i])
            #     for i in range(len(feature_dims))
            # )
            self.n_first_channels = 4
            self.flow_net = ERAFT(n_first_channels=self.n_first_channels)
            # self.flow_net = RAFTSpline()

        if not self.backbone_flag:
        # if True:
            feature_dims = [64, 128, 320, 512]
            # feature_dims = [3]
            self.softsplat_net = Synthesis(feature_dims, activation='PReLU')
            
            # self.MemoryEncoder = nn.ModuleList(
            #     MemoryEncoder(in_dim=feature_dims[i], total_stride=2**i)
            #     for i in range(len(feature_dims))
            # )
            self.MemoryEncoder = MemoryEncoder(in_dim=feature_dims[-1], total_stride=8)
            self.fusion_attens = MultiAttentionBlock(
                                    dim=feature_dims[-1],
                                    num_heads=8,
                                    LayerNorm_type='WithBias',
                                    ffn_expansion_factor=2.66,
                                    bias=False,
                                    is_DA=True
                                )

            # self.fusion_attens = nn.ModuleList(
            #     # Attention(dim=feature_dims[i], num_heads=8, bias=False)
            #     nn.ModuleList(
            #         MultiAttentionBlock(
            #         dim=feature_dims[i],
            #         num_heads=8,
            #         LayerNorm_type='WithBias',
            #         ffn_expansion_factor=2.66,
            #         bias=False,
            #         is_DA=False)
            #         for _ in range(2)
            #     )
            #     for i in range(len(feature_dims))
            # )

        self.apply(self._init_weights)

    def forward(self, x: list, rgb_next: Tensor=None, lookup_timestamps: list=[0.5, 1.0]) -> list:
        feature_init = self.backbone(x)
        y = []
        if len(x) != 1:
            flows_split = []
            tenMetricones = []
            event_voxel = x[1]
            event_voxel_before = x[2]
            # event_voxel_total = x[1]
            # B, C, H, W = event_voxel_total.shape
            # n_it = event_voxel_total.shape[1]//20
            # event_voxel = event_voxel_total.view(-1, 20, H, W)
            if not self.flow_net_flag:
                bin = 5
                event_voxel = torch.cat([event_voxel.mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                ################ raft flow ################
                flow = x[-1]
                # flow = x[2].view(-1, 2, H, W)
                ##########################################

                ################ zero flow ################
                # flow = torch.zeros(event_voxel.shape[0], 2, H, W).to(x[0].device)
                ##########################################

                # tenMetricone = self.softsplat_net(tenEncone=None, tenForward=flow, tenMetricone=None, event_voxel=event_voxel) * 2.0
                # flows_split.append(flow)
                # tenMetricones.append(tenMetricone)

                # feature_after = self.softsplat_net(tenEncone=feature_init, tenForward=flow, tenMetricone=tenMetricone)
                feature_after = self.softsplat_net(tenEncone=feature_init, tenForward=flow, event_voxel=event_voxel)

            else:
                feature_after = feature_init
                # lookup_timestamps = [0.5,1]
                ################ for eraft ################
                # for iter in range(event_voxel.shape[1]//20):
                #     ev = event_voxel[:, iter*20:(iter+1)*20]
                #     bin = 5
                #     ev = torch.cat([ev[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                #     ev1, ev2 = torch.split(ev, ev.shape[1]//2, dim=1)
                #     flow = self.flow_net(ev1, ev2)[-1]
                #     # tenMetricone = self.softsplat_net(tenEncone=None, tenForward=flow, tenMetricone=None, event_voxel=ev) * 2.0
                #     # feature_after = self.softsplat_net(tenEncone=feature_init, tenForward=flow, tenMetricone=tenMetricone)
                #     feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow, event_voxel=ev)
                #     # for i in range(4):
                #     #     for blk in self.fusion_attens[i]:
                #     #         feature_after[i] = blk(feature_after[i], feature_init[i])

                # # one time all version
                # bin = 5
                # ev2 = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                # ev1 = torch.cat([event_voxel_before[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                # flow = self.flow_net(ev1, ev2)[-1]
                # feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow, event_voxel=ev2)
                # # x = self.softsplat_net(tenEncone=[x[0]], tenForward=flow, event_voxel=ev2)
                # # feature_after = self.backbone(x)
                # y_mid = self.decode_head(feature_after)
                # y.append(F.interpolate(y_mid, size=x[0].shape[2:], mode='bilinear', align_corners=False))
                # return y

                # iterative all version

                ##################### eraft memory #################
                mid_supervised = False
                event_voxel_after = x[3]
                bin = 5
                ev_t1_t2 = torch.cat([event_voxel_after[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                ev_t0_t1 = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                ev_before = torch.cat([event_voxel_before[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                flow_t0_t1 = self.flow_net(ev_before, ev_t0_t1)[-1]
                flow_t1_t2 = self.flow_net(ev_t0_t1, ev_t1_t2)[-1]
                if mid_supervised:
                    feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow_t0_t1, event_voxel=ev_t0_t1)
                    y_mid = self.decode_head(feature_after)
                    y_mid = F.interpolate(y_mid, size=x[0].shape[2:], mode='bilinear', align_corners=False)
                    feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow_t1_t2, event_voxel=ev_t1_t2)
                else:
                    # t0 memory
                    ## decode memory
                    y_t0 = self.decode_head(feature_init)
                    # self.memory_bank = [self.MemoryEncoder[i](feature_after[i], y_t0).detach() for i in range(4)]
                    self.memory_bank = self.MemoryEncoder(feature_init[-1], y_t0).detach()

                    # t0 → t1
                    feature_t1 = self.softsplat_net(tenEncone=feature_init, tenForward=flow_t0_t1, event_voxel=ev_t0_t1)
                    ## memory attention Fw, F0_c=None, Kd=None
                    feature_t1[-1] = self.fusion_attens(Fw=feature_t1[-1], F0_c=None, Kd=self.memory_bank)
                    y_t1 = self.decode_head(feature_t1)
                    # self.memory_bank = [self.MemoryEncoder[i](feature_t1[i], y_t1).detach() for i in range(4)]
                    self.memory_bank = self.MemoryEncoder(feature_t1[-1], y_t1).detach()

                    # t1 → t2
                    feature_t2 = self.softsplat_net(tenEncone=feature_t1, tenForward=flow_t1_t2, event_voxel=ev_t1_t2)
                    ## memory attention
                    feature_t2[-1] = self.fusion_attens(Fw=feature_t2[-1], F0_c=None, Kd=self.memory_bank)
                    y_t2 = self.decode_head(feature_t2)
                    y.append(F.interpolate(y_t2, size=x[0].shape[2:], mode='bilinear', align_corners=False))
                    return y

                    # ev = torch.cat([ev_t0_t1, ev_t1_t2], dim=1)
                    # bin = 2
                    # ev = torch.cat([ev[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                    # feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow_t0_t1+flow_t1_t2, event_voxel=ev_t0_t1)
                ##########################################

                # ##################### for eraft ################
                # flows = []
                # ev_before = event_voxel_before[:, -4:]
                # for t in range(5):
                #     ev = event_voxel[:, t*4:(t+1)*4]
                #     flows.append(self.flow_net(ev, ev_before)[-1])
                #     ev_before = ev
                #     flow = sum(flows)
                #     ev_all = event_voxel[:, :(t+1)*4]
                #     bin = t+1
                #     ev_all = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                #     assert ev_all.shape[1] == 4
                #     feature_after = self.softsplat_net(tenEncone=feature_init, tenForward=flow, event_voxel=ev_all)
                #     y_mid = self.decode_head(feature_after)
                #     y.append(F.interpolate(y_mid, size=x[0].shape[2:], mode='bilinear', align_corners=False))
                # return y
            
                ###################### for bflow ################
                # # import ipdb; ipdb.set_trace()
                # lookup_timestamps = [0.2, 0.4, 0.6, 0.8, 1.0]
                # ev = torch.cat([event_voxel_before, event_voxel], dim=1)
                # assert ev.shape[1] == 40
                # bin = event_voxel.shape[1]//10
                # ev = torch.cat([ev[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(10)], dim=1)
                # flow = self.flow_net(ev)[-1]
                # flows = flow.get_flow_from_reference(lookup_timestamps)
                # for t in range(5):
                #     flow = flows[t]
                #     ev_all = event_voxel[:, :(t+1)*4]
                #     bin = t+1
                #     assert ev_all.shape[1] == bin*4
                #     ev_all = torch.cat([ev_all[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                #     feature_after = self.softsplat_net(tenEncone=feature_init, tenForward=flow, event_voxel=ev_all)
                #     y_mid = self.decode_head(feature_after)
                #     y.append(F.interpolate(y_mid, size=x[0].shape[2:], mode='bilinear', align_corners=False))
                # return y

                # mid_supervised = False
                # event_voxel_after = x[3]# bin = 5
                # ev_t1_t2 = torch.cat([event_voxel_after[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                # ev_t0_t1 = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                # ev_before = torch.cat([event_voxel_before[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                # flow_t0_t1 = self.flow_net(ev_before, ev_t0_t1)[-1]
                # flow_t1_t2 = self.flow_net(ev_t0_t1, ev_t1_t2)[-1]
                # if mid_supervised:
                #     feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow_t0_t1, event_voxel=ev_t0_t1)
                #     y_mid = self.decode_head(feature_after)
                #     y.append(F.interpolate(y_mid, size=x[0].shape[2:], mode='bilinear', align_corners=False))
                #     feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow_t1_t2, event_voxel=ev_t1_t2)
                # else:
                #     ev = torch.cat([ev_t0_t1, ev_t1_t2], dim=1)
                #     bin = 2
                #     ev = torch.cat([ev[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                #     feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow_t0_t1+flow_t1_t2, event_voxel=ev_t0_t1)
                ##########################################

                # ################# for bflow ################
                # # 把B C H W -> B C C//2 H W
                # bin = event_voxel.shape[1]//10
                # ev = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(10)], dim=1)
                # flow = self.flow_net(ev)[-1]
                # flows = flow.get_flow_from_reference(lookup_timestamps)

                # for iter in range(len(lookup_timestamps)-1, 0, -1):
                #     flows_split.append(flows[iter] - flows[iter-1])
                # flows_split.append(flows[0])
                # flows_split = flows_split[::-1]

                # # bin = event_voxel.shape[1]//4
                # # ev = torch.cat([ev[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(4)], dim=1)
                # for iter in range(len(lookup_timestamps)):
                #     ev = event_voxel[:, iter*20:(iter+1)*20]
                #     bin = 5
                #     ev = torch.cat([ev[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                #     flow = flows_split[iter]
                #     # tenMetricone = self.softsplat_net(tenEncone=None, tenForward=flow, tenMetricone=None, event_voxel=ev) * 2.0
                #     # feature_after = self.softsplat_net(tenEncone=feature_init, tenForward=flow, tenMetricone=tenMetricone)
                #     feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flow, event_voxel=ev)
                #     # for i in range(4):
                #     #     for blk in self.fusion_attens[i]:
                #     #         feature_after[i] = blk(feature_after[i], feature_init[i])
                # ##########################################
                # # 把B C H W -> B C C//2 H W
                # bin = 2
                # ev = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                # flow = self.flow_net(ev)[-1]
                # flows = flow.get_flow_from_reference(lookup_timestamps)

                # bin = 5
                # event_voxel = torch.cat([event_voxel[:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                # flows_split = []
                # tenMetricones = []
                # for iter in range(len(flows)-1, 0, -1):
                #     flows_split.append(flows[iter] - flows[iter-1])
                #     tenMetricones.append(self.softsplat_net.netSoftmetric(event_voxel, flows[iter] - flows[iter-1]) * 2.0)
                # flows_split.append(flows[0])
                # tenMetricones.append(self.softsplat_net.netSoftmetric(event_voxel, flows[0]) * 2.0)
                # flows_split = flows_split[::-1]
                # tenMetricones = tenMetricones[::-1]
                # ## backbone
                # feature_after = feature_init
                # for iter in range(len(flows_split)):
                #     feature_after = self.softsplat_net(tenEncone=feature_after, tenForward=flows_split[iter], tenMetricone=tenMetricones[iter])
                #     # feature_after = [self.fusion_attens[i](feature_after[i], feature_warp[i]) for i in range(len(feature_warp))]
                #     # for i in range(4):
                #     #     for blk in self.fusion_attens[i]:
                #     #         feature_after[i] = blk(feature_after[i], feature_init[i])

        elif len(x) == 1:
            feature_after = feature_init
            ## decoder
            y_mid = self.decode_head(feature_after)
            y.append(F.interpolate(y_mid, size=x[0].shape[2:], mode='bilinear', align_corners=False))
            return y

        ## visualization
        # import ipdb; ipdb.set_trace()
        # self.visualize_all([x[0]]+feature_before, [rgb_next]+feature_mid, [rgb_next]+feature_after, [flow]+interFlow)
        # self.visualize_features_all(feature_after)
        # self.visualize_all([x[0]]+feature_before, feature_after, [rgb_next]+feature_next, interFlow)
        # exit(0)  


    def visualize_features_all(self, features):
        total_features = sum(f.shape[1] for f in features)
        print(f'Total number of features: {total_features}')
        num_cols = int(total_features ** 0.5)
        num_rows = (total_features + num_cols - 1) // num_cols

        fig = plt.figure(figsize=(100, 60))
        gs = GridSpec(num_rows, num_cols, figure=fig, wspace=0.1, hspace=0.1)

        idx = 0
        for i, feature in enumerate(features):
            # 取第一个 batch 的特征图
            feature_map = feature[0].detach().cpu().numpy()
            for j in range(len(feature_map)):
                if idx >= num_rows * num_cols:
                    break
                h, w = feature_map[j].shape
                ax = fig.add_subplot(gs[idx])
                im = ax.imshow(feature_map[j], cmap='viridis', extent=[0, w, 0, h])
                ax.axis('off')
                idx += 1

        # # 添加颜色条
        # for j in range(num_cols):
        #     img = fig.axes[j].images[0]
        #     fig.colorbar(img, ax=fig.axes[j::num_cols], orientation='vertical', fraction=0.046, pad=0.04)

        plt.savefig('features_all.png', dpi=150)
        plt.show()

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
        # 添加颜色条
        for i in range(4):
            for j in range(num_features):
                img = axes[i, j].images[0]
                fig.colorbar(img, ax=axes[i, j], orientation='vertical', fraction=0.046, pad=0.04)

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
                if self.backbone_flag:
                    msg = self.backbone.load_state_dict(checkpoint, strict=False)
                else:
                    msg = self.load_state_dict(checkpoint, strict=False)
                print("init_pretrained message: ", msg)
    
    def viz2(self, flow, x, rgb_next):
        # 可视化光流
        tenFlow = flow[0].unsqueeze(0)  # .detach().cpu().numpy()
        flow_magnitude = (tenFlow ** 2).sum(axis=0) ** 0.5  # 计算光流的大小
        tenFirst = x[0][0].unsqueeze(0)
        tenSecond = rgb_next[0].unsqueeze(0)
        # end
        # 1 1 H W
        tenMetric_L1 = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenIn=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)
        tenMetric_flow_mag = torch.sqrt(torch.square(tenFlow[:, 0, :, :] + tenFlow[:, 1, :, :])).unsqueeze(1)
        Z = tenMetric_flow_mag

        tenOutputs_softsplat = [softsplat(tenIn=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=-tenMetric_L1, strMode='soft') for fltTime in np.linspace(0.0, 1.0, 11).tolist()]
        npyOutputs_softsplat = [(tenOutput_softsplat[0, :, :, :].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).astype(np.uint8) for tenOutput_softsplat in tenOutputs_softsplat + list(reversed(tenOutputs_softsplat[1:-1]))]
        preds = [np.concatenate([
            # npyOutputs_sum[i][:, :, ::-1],
            # npyOutputs_avg[i][:, :, ::-1],
            # npyOutputs_linear[i][:, :, ::-1], 
            npyOutputs_softsplat[i][:, :, ::-1],
            ], axis=1) for i in range(len(npyOutputs_softsplat))]
        video = moviepy.editor.ImageSequenceClip(sequence=preds, fps=5)
        video.write_gif('./out.gif')

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # 绘制 flow_magnitude
        axes[0, 0].imshow(flow_magnitude[0].cpu().numpy(), cmap='plasma')
        axes[0, 0].set_title('Flow Magnitude')
        axes[0, 0].axis('off')

        # 绘制 tenFirst
        axes[0, 1].imshow(tenFirst[0].cpu().numpy().transpose(1, 2, 0))
        axes[0, 1].set_title('First Frame')
        axes[0, 1].axis('off')

        # 绘制 tenSecond
        axes[1, 0].imshow(tenSecond[0].cpu().numpy().transpose(1, 2, 0))
        axes[1, 0].set_title('Second Frame')
        axes[1, 0].axis('off')

        # 绘制 npyOutputs_softsplat
        axes[1, 1].imshow(npyOutputs_softsplat[0])
        axes[1, 1].set_title('Softsplat Output')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('flow_and_softsplat.png', dpi=150)
        plt.close()

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
