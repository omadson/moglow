import torch
from nflows import transforms

from src.transforms import thops
from ..nets import LSTM


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
    def __init__(self, in_channels, cond_channels, hidden_channels,  network=LSTM):
        super().__init__()
        self.f = network((in_channels // 2)+cond_channels, hidden_channels, 2*(in_channels-in_channels // 2))
        # self.f = network(in_channels, hidden_channels, 2*(in_channels))

    def forward(self, inputs, conds=None, context=None):

        z1, z2 = thops.split_feature(inputs, "split")
        z1_cond = torch.cat((z1, conds), dim=1)
        h = self.f(z1_cond.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        shift, scale = thops.split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2.)+1e-6
        z2 = z2 + shift
        z2 = z2 * scale
        logdet = thops.sum(torch.log(scale), dim=[1, 2])
            
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def inverse(self, inputs, conds=None, context=None):
        z1, z2 = thops.split_feature(inputs, "split")
        z1_cond = torch.cat((z1, conds), dim=1)
        h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
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