import torch
import torch.nn as nn
from typing import Union
from .softsplat import softsplat

class Synthesis(torch.nn.Module):
    """Modified from the synthesis model in Softmax Splatting (https://github.com/sniklaus/softmax-splatting). Modifications:
    1) Warping only one frame with forward flow;
    2) Estimating the importance metric from the input frame and forward flow."""
    
    def __init__(self):
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

        class Warp(torch.nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.nets = nn.ModuleList([
                    Basic('conv-relu-conv', [embed_dim[i] + 1, embed_dim[i], embed_dim[i]], True)
                    for i in range(4)
                ])
            # end

            def forward(self, tenEncone, tenMetricone, tenForward):
                tenOutput = []

                for intLevel in range(4):
                    tenMetricone = torch.nn.functional.interpolate(input=tenMetricone, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                    
                    tenForward = torch.nn.functional.interpolate(input=tenForward, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False) * (float(tenEncone[intLevel].shape[3]) / float(tenForward.shape[3]))
                    
                    tenOutput.append(self.nets[intLevel](
                        softsplat(tenIn=torch.cat([tenEncone[intLevel], tenMetricone], 1), tenFlow=tenForward, tenMetric=tenMetricone, strMode='soft')
                    ))
                # end

                return tenOutput
            # end
        # end

        embed_dim = [64, 128, 320, 512]

        self.netWarp = Warp(embed_dim)

    def forward(self, tenOne, tenEncone, tenForward):
        tenMetricone = torch.sqrt(torch.square(tenForward[:, 0, :, :] + tenForward[:, 1, :, :])).unsqueeze(1)

        tenWarp = self.netWarp(tenEncone, tenMetricone, tenForward)

        return tenWarp

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
