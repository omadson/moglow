import torch
from torch import nn
from nflows import transforms
from nflows.utils import torchutils

from src.transforms import thops
from ..nets import LSTM, LinearZeroInit


def nan_throw(tensor, name="tensor"):
        stop = False
        if ((tensor!=tensor).any()):
            print(name + " has nans")
            stop = True
        if (torch.isinf(tensor).any()):
            print(name + " has infs")
            stop = True
        if stop:
            print(name + ": " + str(tensor))
            #raise ValueError(name + ' contains nans of infs')

class AffineCouplingTransform(transforms.Transform):
    def __init__(
        self,
        in_channels,
        cond_channels,
        hidden_channels,
        network='lstm',
        num_blocks_per_layer=2,
        flow_coupling='additive',
        
    ):
        super().__init__()
        self.network = network
        self.flow_coupling = flow_coupling
        out_channels = (in_channels-in_channels // 2)
        if self.flow_coupling == 'affine':
            out_channels = 2*out_channels
        self.f = LSTM(
            (in_channels // 2)+in_channels,
            hidden_channels,
            out_channels,
            num_blocks_per_layer
        )
        if network.lower() == 'ff':
            self.f = nn.Sequential(
                nn.Linear((in_channels // 2)+cond_channels, hidden_channels),
                nn.ReLU(inplace=False),
                *sum([[nn.Linear(hidden_channels, hidden_channels),nn.ReLU(inplace=False)] for _ in range(num_blocks_per_layer)], []),
                LinearZeroInit(hidden_channels, out_channels)
            ).double()

    def forward(self, inputs, conds=None, context=None, point=False):
        z1, z2 = thops.split_feature(inputs, "split")
        z1_repeated = z1.repeat(1, 1, conds.shape[1]).permute(0, 2, 1)
        z1_cond = torch.cat((z1_repeated, conds), dim=2)
        if self.network.lower() == 'ff':
            z1_cond = torch.cat((z1, conds.flatten(1)[:, :, None]), dim=1).permute(0, 2, 1)
        h = self.f(z1_cond).permute(0, 2, 1)
        shift, scale = thops.split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2.)+1e-6
        z2 = z2 + shift
        z2 = z2 * scale
        logdet = thops.sum(torch.log(scale), dim=[1, 2])
        z = thops.cat_feature(z1, z2)
        if point:
            point_logdet = (
                thops
                .cat_feature(torch.zeros_like(z1), torch.log(scale))
                .squeeze(dim=2)
            )
            return z, logdet, point_logdet
        return z, logdet

    def inverse(self, inputs, conds=None, context=None):
        z1, z2 = thops.split_feature(inputs, "split")
        z1_cond = torch.cat((z1, conds), dim=1)
        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1_cond.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
            logdet = torch.zeros_like(inputs[:, 0, 0])
            z = thops.cat_feature(z1, z2)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
            shift, scale = thops.split_feature(h, "cross")
            nan_throw(shift, "shift")
            nan_throw(scale, "scale")
            nan_throw(z2, "z2 unscaled")
            scale = torch.sigmoid(scale + 2.) + 1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = -thops.sum(torch.log(scale), dim=[1, 2])
        return z, logdet