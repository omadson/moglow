import torch
from torch import nn

from nflows import transforms
from nflows.utils import torchutils

from src.transforms import thops
from src.nets import LSTM, LinearZeroInit


class AffineCouplingTransform(transforms.Transform):
    def __init__(
        self,
        num_features,
        num_conditional_features,
        num_network_layers=2,
        num_neurons_per_layer=128,
        recurrent_network=True,
    ):
        super().__init__()
        out_channels = 2*(num_features-num_features // 2)
        if recurrent_network:
            self.transform_net = LSTM(
                (num_features // 2) + num_conditional_features,
                num_neurons_per_layer,
                out_channels,
                num_network_layers
            )
        else:
            self.transform_net = nn.Sequential(
                nn.Linear((num_features // 2)+num_conditional_features, num_neurons_per_layer),
                nn.ReLU(inplace=False),
                *sum([[nn.Linear(num_neurons_per_layer, num_neurons_per_layer),nn.ReLU(inplace=False)] for _ in range(num_network_layers)], []),
                LinearZeroInit(num_neurons_per_layer, out_channels)
            ).double()
        
    def _coupling_transform_forward(self, inputs, transform_params, point=False):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)

        outputs = inputs * scale + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        if point:
            return outputs, logabsdet, log_scale
        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet
    
    def _scale_and_shift(self, transform_params):
        shift, unconstrained_scale = thops.split_feature(transform_params, "cross")
        scale = torch.sigmoid(unconstrained_scale + 2) + 1e-3
        return scale, shift
    
    def inverse(self, inputs, conds=None, context=None, point=False):
        identity_split, transform_split  = thops.split_feature(inputs, "split")

        identity_split_cond = torch.cat(
            (identity_split, conds),
            dim=2
        )
        transform_params = self.transform_net(
            identity_split_cond
        )

        transform_split, logabsdet = self._coupling_transform_inverse(
            inputs=transform_split,
            transform_params=transform_params
        )
        outputs = thops.cat_feature(identity_split, transform_split)
        return outputs, logabsdet
        
    
    def forward(self, inputs, conds=None, context=None, point=False):
    
        identity_split, transform_split  = thops.split_feature(inputs, "split")
        conds = conds.flatten(start_dim=1)
        identity_split_cond = torch.cat(
            (identity_split, conds),
            dim=1
        )
        transform_params = self.transform_net(
            identity_split_cond
        )
        
        transform_split, *logabsdet = self._coupling_transform_forward(
            inputs=transform_split,
            transform_params=transform_params,
            point=point
        )
        outputs = thops.cat_feature(identity_split, transform_split)
        if point:
            logabsdet, point_logabsdet = logabsdet
            return outputs, logabsdet, torch.cat((point_logabsdet, torch.zeros(identity_split.shape)), dim=1)
        return outputs, logabsdet[0]

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