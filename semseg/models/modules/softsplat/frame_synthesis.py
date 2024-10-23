import torch
import torch.nn as nn
from typing import Union
from .softsplat import softsplat
import numpy as np
from scipy.ndimage import gaussian_filter

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)
    
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (SE Block)"""
    def __init__(self, channels, reduction=4):
        """
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the bottleneck in the SE block.
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        # Two fully connected (1x1 convolution) layers for the squeeze and excitation operations
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global pooling: squeeze operation
        scale = self.avg_pool(x)

        # Excitation operation: FC1 -> ReLU -> FC2 -> Sigmoid
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)

        # Scale input tensor by SE attention
        return x * scale

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the channel attention module.
            kernel_size: Convolution kernel size for the spatial attention module.
        """
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling

        # Shared MLP: FC1 -> ReLU -> FC2 for channel attention
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention Module
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel Attention Module
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention  # Apply channel attention

        # 2. Spatial Attention Module
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))

        return x * spatial_attention  # Apply spatial attention


       



class Synthesis(torch.nn.Module):
    """Modified from the synthesis model in Softmax Splatting (https://github.com/sniklaus/softmax-splatting). Modifications:
    1) Warping only one frame with forward flow;
    2) Estimating the importance metric from the input frame and forward flow."""
    
    def __init__(self, feature_dims, activation='PReLU'):
        super().__init__()

        # 根据传入的激活函数名称选择激活函数，并处理不同的初始化参数
        if activation == 'PReLU':
            # 动态生成 PReLU，后续根据通道数调整
            self.activation_layer = lambda out_channels: nn.PReLU(num_parameters=out_channels, init=0.25)
        elif activation == 'LeakyReLU':
            self.activation_layer = lambda out_channels: nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'ELU':
            self.activation_layer = lambda out_channels: nn.ELU(alpha=1.0)
        elif activation == 'Mish':
            self.activation_layer = lambda out_channels: nn.Mish()
        elif activation == 'ReLU':
            self.activation_layer = lambda out_channels: nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        class Basic(torch.nn.Module):
            def __init__(self, strType, intChannels, boolSkip, skip_type='residual', attention_type='cbam', activation_layer=None):
                super().__init__()
                self.activation_layer = activation_layer
                if strType == 'relu-conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        self.activation_layer(intChannels[0]),
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, padding=1, bias=False)
                    )
                elif strType == 'conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-more-more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//8, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//8),
                        torch.nn.Conv2d(in_channels=intChannels[1]//8, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                # end

                self.boolSkip = boolSkip

                if boolSkip == True:
                    if intChannels[0] == intChannels[2]:
                        self.netShortcut = None

                    elif intChannels[0] != intChannels[2]:
                        self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)

                self.skip_type = skip_type
                self.attention_type = attention_type
                # 使用一个卷积层来确保不同跳跃连接输出的形状与预期一致
                if self.skip_type == 'dense':
                    self.matchChannels = torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)
                elif self.skip_type == 'attention':
                    if self.attention_type == 'se':
                        self.se_block = SEBlock(intChannels[2])
                    elif self.attention_type == 'cbam':
                        self.cbam_block = CBAM(intChannels[2])

                    # end
                # end
            # end


            def forward(self, tenInput):
                # Standard path through the main network
                tenMain = self.netMain(tenInput)

                if self.boolSkip == False:
                    return tenMain

                if self.skip_type == 'residual':
                    # Residual connections: output shape matches input shape
                    if self.netShortcut is None:
                        return tenMain + tenInput
                    else:
                        return tenMain + self.netShortcut(tenInput)

                elif self.skip_type == 'dense':
                    # Dense connections: concatenate input and output
                    return self.dense_forward(tenInput)

                elif self.skip_type == 'attention':
                    # Attention-based skip connections
                    if self.attention_type == 'se':
                        if self.netShortcut is None:
                            return self.se_block(tenMain) + tenInput
                        else:
                            return self.se_block(tenMain) + self.netShortcut(tenInput)
                    elif self.attention_type == 'cbam':
                        if self.netShortcut is None:
                            return self.cbam_block(tenMain) + tenInput
                        else:
                            return self.cbam_block(tenMain) + self.netShortcut(tenInput)

            def dense_forward(self, tenInput):
                """ Dense forward pass with DenseNet-style concatenation. """
                features = [tenInput]  # Initialize the feature list with the input
                for layer in self.netMain:
                    new_feature = layer(torch.cat(features, dim=1))  # Concatenate all previous features
                    features.append(new_feature)
                # After passing through all layers, concatenate all features
                print(torch.cat(features, dim=1).shape)
                return self.matchChannels(torch.cat(features, dim=1))

                # end
            # end
        # end

        class Downsample(torch.nn.Module):
            def __init__(self, intChannels, activation_layer=None):
                super().__init__()
                self.activation_layer = activation_layer

                self.netMain = torch.nn.Sequential(
                    self.activation_layer(intChannels[0]),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Upsample(torch.nn.Module):
            def __init__(self, intChannels, activation_layer=None):
                super().__init__()
                self.activation_layer = activation_layer
                self.netMain = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    self.activation_layer(intChannels[0]),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Upsample_CrossScale(torch.nn.Module):
            def __init__(self, intChannels, upsample_mode='bilinear'):
                super().__init__()
                
                # 调整高分辨率特征图的通道数
                self.conv_high_res = torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[0], kernel_size=1, stride=1, padding=0, bias=False)
                # 上采样层，将低分辨率特征图上采样到与高分辨率特征图相同的空间尺寸
                self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
                self.netMain = torch.nn.Sequential(
                    # torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    self.activation_layer(intChannels[0]*2),
                    torch.nn.Conv2d(in_channels=intChannels[0]*2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, high_res_feature, low_res_feature):
                # 对高分辨率特征图应用 1x1 卷积，调整通道数
                high_res_feature = self.conv_high_res(high_res_feature)

                # 对低分辨率特征图进行上采样
                low_res_feature_upsampled = self.upsample(low_res_feature)

                tenInput = torch.cat([high_res_feature, low_res_feature_upsampled], 1)
                return self.netMain(tenInput)
            # end
        # end

        class Softmetric(torch.nn.Module):
            def __init__(self, skip_type, in_ch, out_ch, attention_type=None, activation_layer=None):
                super().__init__()
                # embed_dim = [32, 64, 128, 256]
                embed_dim = [16, 32, 64, 96]

                # self.netEventInput = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
                self.netEventInput = torch.nn.Conv2d(in_channels=in_ch[0], out_channels=embed_dim[0]//2, kernel_size=3, stride=1, padding=1, bias=False)
                # self.netRGBInput = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
                self.netFlow = torch.nn.Conv2d(in_channels=in_ch[1], out_channels=embed_dim[0]//2, kernel_size=3, stride=1, padding=1, bias=False)
                for intRow, intFeatures in enumerate(embed_dim):
                    self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True, skip_type, attention_type, activation_layer))
                # end

                for intCol in [0]:
                    self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([embed_dim[0], embed_dim[1], embed_dim[1]], activation_layer=activation_layer))
                    self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([embed_dim[1], embed_dim[2], embed_dim[2]], activation_layer=activation_layer))
                    self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([embed_dim[2], embed_dim[3], embed_dim[3]], activation_layer=activation_layer))
                # end

                for intCol in [1]:
                    self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([embed_dim[3], embed_dim[2], embed_dim[2]], activation_layer=activation_layer))
                    self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([embed_dim[2], embed_dim[1], embed_dim[1]], activation_layer=activation_layer))
                    self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([embed_dim[1], embed_dim[0], embed_dim[0]], activation_layer=activation_layer))
                    # self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample_CrossScale([embed_dim[3], embed_dim[2], embed_dim[2]]))
                    # self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample_CrossScale([embed_dim[2], embed_dim[1], embed_dim[1]]))
                    # self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample_CrossScale([embed_dim[1], embed_dim[0], embed_dim[0]]))
                # end

                self.netOutput = Basic('conv-relu-conv', [embed_dim[0], embed_dim[0], out_ch], True, activation_layer=activation_layer)
                # self.netOutput = Basic('more-more-conv', [embed_dim[0], embed_dim[0], 1], True)
            # end

            def forward(self, in1, in2):
                tenColumn = [None, None, None, None]

                tenColumn[0] = torch.cat([
                    self.netEventInput(in1),
                    self.netFlow(in2),
                ], 1)
                # tenColumn[0]  = self.netEventInput(event_voxel)
                tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
                tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
                tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])

                # skip residual
                intColumn = 1
                for intRow in range(len(tenColumn) -1, -1, -1):
                    # '3x0 - 3x1'
                    tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
                    if intRow != len(tenColumn) - 1:
                        tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])
                        # if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                        # if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                        tenColumn[intRow] = tenColumn[intRow] + tenUp
                    # end
                # end

                # # cross-connections
                # intColumn = 1
                # tenColumn[3] = self._modules['3x0 - 3x1'](tenColumn[3])
                # for intRow in range(len(tenColumn) -2, -1, -1):
                #     tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
                #     if intRow != 0:
                #         tenColumn[intRow] = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow], tenColumn[intRow+1])
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
            def __init__(self, embed_dim, activation_layer=None):
                super().__init__()
                self.nets = nn.ModuleList([
                    # nn.Sequential(
                    # Basic('conv-relu-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True)
                    # Basic('more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True)
                    # Basic('more-more-more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True)
                    Basic('more-more-conv', [embed_dim[i]+1, embed_dim[i], embed_dim[i]], True, activation_layer=activation_layer)
                    # SELayer(embed_dim[i]),
                    # )
                    for i in range(len(embed_dim))
                ])
            # end

            def forward(self, tenEncone, tenMetricone, tenForward, tenScale=None):
                tenOutput = []
                tenMid = []
                tenFlow = []

                for intLevel in range(len(tenEncone)):
                    tenMetricone = torch.nn.functional.interpolate(input=tenMetricone, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                    
                    tenForward = torch.nn.functional.interpolate(input=tenForward, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False) * (float(tenEncone[intLevel].shape[3]) / float(tenForward.shape[3]))
                    # tenScale = torch.nn.functional.interpolate(input=tenScale, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
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
                    tenMid.append(tenWarp)
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

                return tenOutput, tenMid, tenFlow
            # end
        # end
        # skip_type = 'residual'
        # skip_type = 'dense'
        skip_type = 'attention'

        # attention_type = 'se'
        attention_type = 'cbam'

        # self.netFlow = Softmetric(skip_type, in_ch=[4,3], out_ch=2, attention_type=attention_type, activation_layer=self.activation_layer)
        self.netSoftmetric = Softmetric(skip_type, in_ch=[4,2], out_ch=1, attention_type=attention_type, activation_layer=self.activation_layer)
        # self.netScale = Softmetric()

        self.netWarp = Warp(feature_dims, activation_layer=self.activation_layer)

        # # 定义可学习的alpha参数，并初始化为1
        # self.alpha_s = nn.Parameter(torch.tensor(2.0))
        # self.alpha_s_p = nn.Parameter(torch.tensor(1.0))
        # self.alpha_s_f = nn.Parameter(torch.tensor(1.0))
        # self.alpha_s_v = nn.Parameter(torch.tensor(1.0))

        # self.alpha_m_p = nn.Parameter(torch.tensor(1.0))
        # self.alpha_m_f = nn.Parameter(torch.tensor(1.0))
        # self.alpha_m_v = nn.Parameter(torch.tensor(1.0))


    def forward(self, tenEncone, rgb, event_voxel, tenForward):
    # def forward(self, tenEncone, tenForward, event_voxel, tenEncone_event=None, psi=None):
        # tenMetricone = torch.sqrt(torch.square(tenForward[:, 0, :, :] + tenForward[:, 1, :, :])).unsqueeze(1)
        # tenForward = self.netFlow(event_voxel, rgb) * 2.0
        tenMetricone = self.netSoftmetric(event_voxel, tenForward) * 2.0
        # tenMetricone = self.netSoftmetric(rgb, event_voxel, tenForward) * self.alpha_s
        # tenMetricone, tenScale = torch.chunk(self.netSoftmetric(rgb, event_voxel, tenForward) * 2.0, chunks=2, dim=1)
        tenWarp, tenMid, tenFlow = self.netWarp(tenEncone, tenMetricone, tenForward)
        # tenWarp, tenFlow = self.netWarp(tenEncone, tenMetricone, tenForward, tenScale)
        # tenScale = self.netScale(rgb, event_voxel, tenForward)
        # # element-wise multiplication
        # print(tenWarp.shape, tenScale.shape)
        # tenWarp = tenWarp * tenScale
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

        return tenWarp, tenMid, tenFlow

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