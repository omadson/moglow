import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from nflows import transforms, flows, distributions

from .flow import Flow
from .transforms import (
    thops,
    CompositeTransform,
    ActNorm2d,
    InvertibleConv1x1,
    AffineCouplingTransform,
    ReshapeTransform
)


class Moglow(Flow):
    def __init__(
        self,
        features,
        conditional_features,
        sequence_length,
        num_layers=3,
        coupling_flow='affine',
        coupling_network='LSTM',
        hidden_features=128,
        num_blocks_per_layer=2
    ):
        self.num_blocks_per_layer = num_blocks_per_layer
        self.hidden_features = hidden_features

        layers = []
        for _ in range(num_layers):
            layers.append(CompositeTransform([
                # 1. actnorm
                ActNorm2d(features), 
                # 2. permute
                InvertibleConv1x1(features, LU_decomposed=True), 
                # 3. coupling
                AffineCouplingTransform(
                    in_channels=features,
                    cond_channels=conditional_features,
                    hidden_channels=hidden_features,
                    network=coupling_network,
                    num_blocks_per_layer=num_blocks_per_layer,
                    flow_coupling=coupling_flow
                ) 
            ]))
        
        layers.append(ReshapeTransform(
            input_shape = (features, sequence_length),
            output_shape = (features * sequence_length,)
        ))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=distributions.StandardNormal((features * sequence_length,)),
        )
        
    def init_lstm_hidden(self, inputs):
        for step_transforms in self._transform.children():
            for step_transform in step_transforms:
                for transform in step_transform.children():
                    for inner_transform in transform:
                        if isinstance(inner_transform, AffineCouplingTransform):
                            if inner_transform.network.lower() == 'lstm':
                                inner_transform.f.init_hidden(inputs['x'].shape[0])
                                z1, z2 = thops.split_feature(inputs['x'], "split")
                                z1_cond = torch.cat((z1, inputs['cond']), dim=1)  
                                inner_transform.f(z1_cond.permute(0, 2, 1))
    
    def repackage_lstm_hidden(self):
        for step_transforms in self._transform.children():
            for step_transform in step_transforms:
                for transform in step_transform.children():
                    for inner_transform in transform:
                        if isinstance(inner_transform, AffineCouplingTransform):
                            if inner_transform.network.lower() == 'lstm':
                                inner_transform.f.hidden = tuple(
                                    Variable(v.data) for v in inner_transform.f.hidden
                                )
                                # inner_transform.f.init_hidden(inputs['x'].shape[0])
                                # z1, z2 = thops.split_feature(inputs['x'], "split")
                                # z1_cond = torch.cat((z1, inputs['cond']), dim=1)  
                                # inner_transform.f(z1_cond.permute(0, 2, 1))
                            