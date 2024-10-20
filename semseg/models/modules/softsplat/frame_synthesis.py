import torch
import torch.nn as nn
from typing import Union
from .softsplat import softsplat
import numpy as np
from scipy.ndimage import gaussian_filter

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class Synthesis(torch.nn.Module):
    """Modified from the synthesis model in Softmax Splatting (https://github.com/sniklaus/softmax-splatting). Modifications:
    1) Warping only one frame with forward flow;
    2) Estimating the importance metric from the input frame and forward flow."""
    
    def __init__(self, feature_dims):
        super().__init__()

        class Basic(torch.nn.Module):
            def __init__(self, strType, intChannels, boolSkip):
                super().__init__()

                if strType == 'relu-conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )

                elif strType == 'conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//2, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//2, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//4, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//2, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-more-more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//2, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//4, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//8, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//8, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//8, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//4, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1]//2, init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                # end

                self.boolSkip = boolSkip

                if boolSkip == True:
                    if intChannels[0] == intChannels[2]:
                        self.netShortcut = None

                    elif intChannels[0] != intChannels[2]:
                        self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)

                    # end
                # end
            # end

            def forward(self, tenInput):
                if self.boolSkip == False:
                    return self.netMain(tenInput)
                # end

                if self.netShortcut is None:
                    return self.netMain(tenInput) + tenInput

                elif self.netShortcut is not None:
                    return self.netMain(tenInput) + self.netShortcut(tenInput)

                # end
            # end
        # end

        class Downsample(torch.nn.Module):
            def __init__(self, intChannels):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Upsample(torch.nn.Module):
            def __init__(self, intChannels):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Softmetric(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # self.netEventInput = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
                self.netEventInput = torch.nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
                # self.netRGBInput = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False)
                self.netFlow = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)

                for intRow, intFeatures in [(0, 16), (1, 32), (2, 64), (3, 96)]:
                    self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
                # end

                for intCol in [0]:
                    self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([16, 32, 32]))
                    self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([32, 64, 64]))
                    self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([64, 96, 96]))
                # end

                for intCol in [1]:
                    self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([96, 64, 64]))
                    self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([64, 32, 32]))
                    self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([32, 16, 16]))
                # end

                self.netOutput = Basic('conv-relu-conv', [16, 16, 1], True)
            # end

            def forward(self, event_voxel, tenFlow):
                tenColumn = [None, None, None, None]

                tenColumn[0] = torch.cat([
                    self.netEventInput(event_voxel),
                    self.netFlow(tenFlow),
                ], 1)
                tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
                tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
                tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])

                intColumn = 1
                for intRow in range(len(tenColumn) -1, -1, -1):
                    tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
                    if intRow != len(tenColumn) - 1:
                        tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                        if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                        if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                        tenColumn[intRow] = tenColumn[intRow] + tenUp
                    # end
                # end

                return self.netOutput(tenColumn[0])
                # return torch.sigmoid(self.netOutput(tenColumn[0]))
            # end
        # end

        # class Warp(torch.nn.Module):
        #     def __init__(self, embed_dim):
        #         super().__init__()
        #         self.nets = nn.ModuleList([
        #             Basic('conv-relu-conv', [embed_dim[i], embed_dim[i], embed_dim[i]], True)
        #             for i in range(len(embed_dim))
        #         ])
        #     # end

        #     def forward(self, tenEncone, tenMetricone_splat, tenMetricone_merge, tenForward, tenEncone_event):
        #     # def forward(self, tenEncone, tenMetricone, tenForward, tenEncone_event):
        #         tenOutput = []
                
        #         # Ft_0 = softsplat(tenIn=torch.cat([tenMetricone_merge, -tenForward], 1), tenFlow=tenForward, tenMetric=tenMetricone_splat, strMode='soft')
        #         Ft_0 = softsplat(tenIn=-tenForward, tenFlow=tenForward, tenMetric=tenMetricone_splat, strMode='soft')

        #         for intLevel in range(len(tenEncone)):
        #             Ft_0 = torch.nn.functional.interpolate(input=Ft_0, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
        #             # M_splat = torch.nn.functional.interpolate(input=tenMetricone_splat, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
        #             tenWarp = backwarp(tenIn=tenEncone[intLevel], tenFlow=Ft_0)
        #             # tenWarp = backwarp(tenIn=torch.cat([tenEncone[intLevel], M_splat], 1), tenFlow=Ft_0)
        #             tenOutput.append(self.nets[intLevel](
        #                 # torch.cat([tenEncone[intLevel], tenEncone_event[intLevel], tenWarp], 1)
        #                 # torch.cat([tenEncone[intLevel], tenWarp], 1)
        #                 tenWarp
        #                 # tenWarp+tenIn
        #                 # tenWarp + tenEncone[intLevel]
        #                 # tenWarp + tenEncone_event[intLevel]
        #                 # tenWarp + tenEncone[intLevel] + tenEncone_event[intLevel]
        #             ))
        #         # end

        #         return tenOutput, None
        #     # end
        # # end

        class Warp(torch.nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.nets = nn.ModuleList([nn.Sequential(
                    # Basic('conv-relu-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True)
                    # Basic('more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True)
                    Basic('more-more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True),
                    SELayer(embed_dim[i]),
                    # Basic('more-more-more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True)
                    )for i in range(len(embed_dim))
                ])
            # end

            def forward(self, tenEncone, tenMetricone, tenForward):
                tenOutput = []
                tenFlow = []

                for intLevel in range(len(tenEncone)):
                    tenMetricone = torch.nn.functional.interpolate(input=tenMetricone, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                    
                    tenForward = torch.nn.functional.interpolate(input=tenForward, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False) * (float(tenEncone[intLevel].shape[3]) / float(tenForward.shape[3]))
                    tenFlow.append(tenForward)
                    tenIn=torch.cat([tenEncone[intLevel], tenMetricone], 1)
                    # tenMask = torch.ones_like(tenMetricone)
                    # tenIn = tenEncone[intLevel]
                    tenWarp = softsplat(tenIn=tenIn, tenFlow=tenForward, tenMetric=tenMetricone, strMode='soft')
                    # tenMaskWarp = softsplat(tenIn=tenMask, tenFlow=tenForward, tenMetric=tenMetricone, strMode='soft')
                    # tenMaskWarp = tenMaskWarp.expand(-1, tenIn.shape[1], -1, -1)
                    # print(tenWarp.shape, tenMaskWarp.shape, tenIn.shape)
                    # print((tenMaskWarp > 0).shape)
                    # tenWarp = tenWarp[tenMaskWarp > 0] + tenIn[tenMaskWarp == 0]
                    tenOutput.append(self.nets[intLevel](
                        # torch.cat([tenEncone[intLevel], tenEncone_event[intLevel], tenWarp], 1)
                        # torch.cat([tenEncone[intLevel], tenWarp], 1)
                        tenWarp
                        # tenWarp+tenIn
                        # tenWarp + tenEncone[intLevel]
                        # tenWarp + tenEncone_event[intLevel]
                        # tenWarp + tenEncone[intLevel] + tenEncone_event[intLevel]
                    ))
                # end

                return tenOutput, tenFlow
            # end
        # end

        self.netSoftmetric = Softmetric()

        self.netWarp = Warp(feature_dims)

        # # 定义可学习的alpha参数，并初始化为1
        # self.alpha_s = nn.Parameter(torch.tensor(1.0))
        # self.alpha_s_p = nn.Parameter(torch.tensor(1.0))
        # self.alpha_s_f = nn.Parameter(torch.tensor(1.0))
        # self.alpha_s_v = nn.Parameter(torch.tensor(1.0))

        # self.alpha_m_p = nn.Parameter(torch.tensor(1.0))
        # self.alpha_m_f = nn.Parameter(torch.tensor(1.0))
        # self.alpha_m_v = nn.Parameter(torch.tensor(1.0))


    def forward(self, tenEncone, event_voxel, tenForward):
    # def forward(self, tenEncone, tenForward, event_voxel, tenEncone_event=None, psi=None):
        # tenMetricone = torch.sqrt(torch.square(tenForward[:, 0, :, :] + tenForward[:, 1, :, :])).unsqueeze(1)
        # tenMetricone = self.netSoftmetric(rgb, event_voxel, tenForward) * 2.0
        tenMetricone = self.netSoftmetric(event_voxel, tenForward) * 2.0
        tenWarp, tenFlow = self.netWarp(tenEncone, tenMetricone, tenForward)
        # tenMetricone_splat, tenMetricone_merge = torch.chunk(self.netSoftmetric(event_voxel, tenForward) * 2.0, chunks=2, dim=1)

        # # 计算各个度量
        # psi_photo = psi[:,0].unsqueeze(1)
        # psi_flow = psi[:,1].unsqueeze(1)
        # psi_varia = psi[:,2].unsqueeze(1)
        
        # # 计算 Splatting 度量标准
        # tenMetricone_splat = (1 / (1 + (self.alpha_s_p * psi_photo))) + \
        #           (1 / (1 + (self.alpha_s_f * psi_flow))) + \
        #           (1 / (1 + (self.alpha_s_v * psi_varia)))
        # tenMetricone_splat = self.alpha_s * tenMetricone_splat

        # tenMetricone_merge = (1 / (1 + (self.alpha_m_p * psi_photo))) + \
        #             (1 / (1 + (self.alpha_m_f * psi_flow))) + \
        #             (1 / (1 + (self.alpha_m_v * psi_varia)))

        # tenWarp, tenFlow = self.netWarp(tenEncone, tenMetricone_splat, tenMetricone_merge, tenForward, tenEncone_event)

        return tenWarp, tenFlow

@torch.no_grad()
def predict_tensor(src_frame: torch.Tensor, flow: torch.Tensor, model: Synthesis, batch_size: int = 32):
    # src_frame and flow should be normalized tensors
    out_frames = []
    for i in range(0, flow.shape[0], batch_size):
        bs = min(batch_size, flow.shape[0] - i)
        out_frames.append(model(src_frame.repeat(bs, 1, 1, 1), flow[i:i + bs]))
    out_frames = torch.cat(out_frames, dim=0)
    
    return out_frames

@torch.no_grad()
def softsplat_tensor(src_frame: torch.Tensor, flow: torch.Tensor, transforms, weight_type: Union[None, str] = None, return_tensor: bool = True):
    # src_frame and flow should be normalized tensors
    if weight_type == "flow_mag":
        weight = torch.sqrt(torch.square(flow[:, 0, :, :] + flow[:, 1, :, :])).unsqueeze(1)
        mode = "soft"
    else:
        weight = None
        mode = "avg"
    out_frames = softsplat(tenIn=src_frame.repeat(flow.shape[0], 1, 1, 1), tenFlow=flow, tenMetric=weight, strMode=mode)
    if return_tensor:
        return transforms.denormalize_frame(out_frames)
    else:
        return transforms.deprocess_frame(out_frames)

def backwarp(tenIn, tenFlow):
    tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[3], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
    tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[2], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

    tenGrid = torch.cat([tenHor, tenVer], 1).cuda()

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(tenGrid + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
# end

def compute_photometric_consistency(I0, I1, F0to1):
    """计算光度一致性 ψ_photo"""
    warped_I1 = backwarp(I1, F0to1)
    psi_photo = np.abs(I0 - warped_I1)
    return psi_photo

def compute_flow_consistency(F0to1, F1to0):
    """计算光流一致性 ψ_flow"""
    # 反向映射光流 F1to0
    warped_F1to0_x = backwarp(F1to0[..., 0], F0to1)
    warped_F1to0_y = backwarp(F1to0[..., 1], F0to1)
    # 计算一致性
    psi_flow = np.sqrt((F0to1[..., 0] + warped_F1to0_x)**2 + (F0to1[..., 1] + warped_F1to0_y)**2)
    return psi_flow

def compute_flow_variance(F0to1):
    """计算光流方差 ψ_varia"""
    F_squared = F0to1 ** 2
    G_F_squared_x = gaussian_filter(F_squared[..., 0], sigma=1)
    G_F_squared_y = gaussian_filter(F_squared[..., 1], sigma=1)
    
    G_F_x = gaussian_filter(F0to1[..., 0], sigma=1)
    G_F_y = gaussian_filter(F0to1[..., 1], sigma=1)
    
    variance_x = G_F_squared_x - (G_F_x ** 2)
    variance_y = G_F_squared_y - (G_F_y ** 2)
    
    psi_varia = np.sqrt(variance_x + variance_y)
    return psi_varia

def compute_metric(psi_photo, psi_flow, psi_varia, alpha_p, alpha_f, alpha_v):
    """计算度量标准 M"""
    M = (1 / (1 + (alpha_p * psi_photo))) + \
        (1 / (1 + (alpha_f * psi_flow))) + \
        (1 / (1 + (alpha_v * psi_varia)))
    return M