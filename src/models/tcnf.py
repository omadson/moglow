from enum import Enum

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pydantic import BaseModel, PositiveInt, conint

from ..flow import Flow
from ..distribuitions import StandardNormal
from ..transforms import CompositeTransform
from ..transforms.conditional.coupling import AffineCouplingTransform
from ..transforms.conditional.permutations import RandomPermutation
from ..transforms.conditional.normalization import ActNorm


class TCNFConfig(BaseModel):
    num_features: PositiveInt
    num_flows: conint(gt=0, lt=30) = 3
    num_network_layers: conint(gt=0, lt=30) = 2
    num_neurons_per_layer: conint(gt=2, lt=2**10) = 2**5
    recurrent_network: bool = True

class TCNF(Flow):
    def __init__(
            self,
            num_features,
            num_flows=3,
            # coupling
            num_network_layers=2,
            num_neurons_per_layer=128,
            recurrent_network=True,
        ):
        layers = []
        self.num_features = num_features
        self.num_flows = num_flows
        self.num_network_layers = num_network_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.recurrent_network = recurrent_network
        for _ in range(num_flows):
            layers.extend([
                # 1. actnorm
                ActNorm(num_features),
                # 2. permute
                RandomPermutation(num_features),
                # 3. coupling
                AffineCouplingTransform(
                    input_length=num_features,
                    num_network_layers=num_network_layers,
                    num_neurons_per_layer=num_neurons_per_layer,
                    recurrent_network=recurrent_network,
                )
            ])

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal((num_features,)),
        )

    def init_lstm_hidden(self):
        for transform in next(self._transform.children()):
            if transform._get_name() == 'AffineCouplingTransform' and self.recurrent_network:
                transform.transform_net.init_hidden()
                
    def repackage_lstm_hidden(self):
        for transform in next(self._transform.children()):
            if transform._get_name() == 'AffineCouplingTransform' and self.recurrent_network:
                transform.transform_net.hidden = tuple(Variable(v.data) for v in transform.transform_net.hidden)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TCNFTrainer:
    @staticmethod
    def create(dataset_info, config, device):
        return TCNF(
            **TCNFConfig(
                **dataset_info,
                **config
            ).dict()
        ).to(device)
    
    @staticmethod
    def train(model, optimizer, scheduler, train_loader, device=None):
        device = device or torch.device("cpu")
        running_loss = []
        model.train()
        for i, data in enumerate(train_loader):
            inputs = data['x'].to(device).squeeze(dim=2)
            conds = data['cond'].to(device).squeeze(dim=1)
            if i == 0:
                model.init_lstm_hidden()
            else:
                model.repackage_lstm_hidden()
            optimizer.zero_grad()
            loss = -model.log_prob(inputs=inputs, conds=conds).mean()
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
                conds = data['cond'].to(device).squeeze(dim=1)
                model.init_lstm_hidden()
                loss_valid = -model.log_prob(inputs=inputs, conds=conds).mean()
                valid_losses.append(loss_valid.cpu().item())
        return np.average(valid_losses)
    
    @staticmethod
    def get_scores(model, data_set, device, point=False, init_lstm_hidden=True):
        batch_size = 1
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size
        )
        model.eval()
        with torch.no_grad():
            log_prob = []
            for i, data_batch in enumerate(data_loader):
                if i == 0 and init_lstm_hidden:
                    model.init_lstm_hidden()
                log_prob.append(
                    -model
                    .log_prob(
                        inputs=data_batch['x'].to(device).squeeze(dim=2),
                        conds=data_batch['cond'].to(device).squeeze(dim=1),
                        point=point
                    )
                )
        return torch.cat(log_prob, dim=0).detach().numpy()

    