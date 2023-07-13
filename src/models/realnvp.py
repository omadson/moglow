from enum import Enum

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pydantic import BaseModel, PositiveInt, conint, confloat

import torch
from torch.nn import functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm
from ..nets import ResidualNet



class RealNVPConfig(BaseModel):
    num_features: PositiveInt
    num_flows: conint(gt=0, lt=30) = 3
    num_network_layers: conint(gt=0, lt=30) = 2
    num_neurons_per_layer: conint(gt=2, lt=2**10) = 2**5
    use_volume_preserving: bool = True
    dropout_probability: confloat(ge=0.0, lt=1.0) = 0.0,
    batch_norm_within_layers: bool = False
    batch_norm_between_layers: bool = False

class RealNVP(Flow):
    """An simplified version of Real NVP for 1-dim inputs.

    This implementation uses 1-dim checkerboard masking but doesn't use multi-scaling.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(
        self,
        num_features,
        num_neurons_per_layer,
        num_flows,
        num_network_layers,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(num_features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return ResidualNet(
                in_features,
                out_features,
                hidden_features=num_neurons_per_layer,
                num_blocks=num_network_layers,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_flows):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=num_features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([num_features]),
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RealNVPTrainer:
    @staticmethod
    def create(dataset_info, config, device):
        return RealNVP(
            **RealNVPConfig(
                **dataset_info,
                **config
            ).dict()
        ).to(device)
    
    @staticmethod
    def train(model, optimizer, scheduler, train_loader, device=None):
        device = device or torch.device("cpu")
        running_loss = []
        model.train()
        for data in train_loader:
            inputs = data['x'].to(device).squeeze(dim=2)
            loss = -model.log_prob(inputs=inputs).mean()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().item())
        scheduler.step()
        return np.average(running_loss)
    
    @staticmethod
    def validation(model, valid_loader, device):
        device = device or torch.device("cpu")
        valid_losses = []
        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                inputs = data['x'].to(device).squeeze(dim=2)
                loss_valid = -model.log_prob(inputs=inputs).mean()
                valid_losses.append(loss_valid.cpu().item())
        return np.average(valid_losses)
    
    @staticmethod
    def get_scores(model, data_set, device, point=False, init_lstm_hidden=True):
        batch_size = 128
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size
        )
        model.eval()
        with torch.no_grad():
            log_prob = []
            for i, data_batch in enumerate(data_loader):
                data_x = data_batch['x'].to(device).squeeze(dim=2)
                log_prob.append(
                    -model
                    .log_prob(
                        inputs=data_x
                    ).repeat(data_x.shape[1], 1).T
                )
        return torch.cat(log_prob, dim=0).detach().numpy()

    