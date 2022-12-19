import torch
from torch import optim
from torch.utils.data import DataLoader
from nflows import transforms, flows, distributions

from .flow import Flow
from .transforms import (
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
        num_blocks_per_layer=128,
    ):

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
                    hidden_channels=num_blocks_per_layer,
                    network=coupling_network,
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
