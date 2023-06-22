from enum import Enum

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pydantic import BaseModel, PositiveInt, conint

from ..flow import Flow
from ..distribuitions import StandardNormal
from ..transforms import (
    CompositeTransform,
    ActNorm2d,
    InvertibleConv1x1,
    AffineCouplingTransform,
)


class CouplingFlowTypes(str, Enum):
    affine = "affine"
    additive = "additive"


class CouplingNetworkTypes(str, Enum):
    ff = 'ff'
    lstm = 'lstm'
    gru = 'gru'


class MoglowConfig(BaseModel):
    num_features: PositiveInt
    num_conditional_features: PositiveInt
    sequence_length: PositiveInt = 3
    num_layers: conint(gt=0, lt=30) = 3
    coupling_flow: CouplingFlowTypes = CouplingFlowTypes.affine
    coupling_network: CouplingNetworkTypes = CouplingNetworkTypes.ff
    num_hidden_features: conint(gt=2, lt=2**10) = 2**5
    num_hidden_blocks: conint(gt=0, lt=30) = 2


class Moglow(Flow):
    def __init__(
        self,
        num_features,
        num_conditional_features,
        sequence_length,
        num_layers=3,
        coupling_flow='affine',
        coupling_network='LSTM',
        num_hidden_features=128,
        num_hidden_blocks=2,
    ):
        self.num_blocks_per_layer = num_hidden_blocks
        self.num_hidden_features = num_hidden_features
        self.name = (
            f"moglow_{num_layers}_{coupling_flow.value}_{coupling_network.value}_"
            f"{num_hidden_blocks}_{num_hidden_features}"
        )
        self.coupling_network = coupling_network
        layers = []
        for _ in range(num_layers):
            layers.extend([
                # 1. actnorm
                ActNorm2d(num_features), 
                # 2. permute
                InvertibleConv1x1(num_features, LU_decomposed=True), 
                # 3. coupling
                AffineCouplingTransform(
                    in_channels=num_features,
                    cond_channels=num_conditional_features,
                    hidden_channels=num_hidden_features,
                    network=coupling_network,
                    num_blocks_per_layer=num_hidden_blocks,
                    flow_coupling=coupling_flow
                ) 
            ])

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal((num_features, sequence_length,)),
        )
        
    def init_lstm_hidden(self):
        for transform in next(self._transform.children()):
            if transform._get_name() == 'AffineCouplingTransform' and self.coupling_network.lower() == 'lstm':
                transform.f.init_hidden()
                
    def repackage_lstm_hidden(self):
        for transform in next(self._transform.children()):
            if transform._get_name() == 'AffineCouplingTransform' and self.coupling_network.lower() == 'lstm':
                transform.f.hidden = tuple(Variable(v.data) for v in transform.f.hidden)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MoglowTrainer:
    @staticmethod
    def create(dataset_info, config, device):
        return Moglow(
            **MoglowConfig(
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
            inputs = data['x'].to(device)
            conds = data['cond'].to(device)
            model.init_lstm_hidden()
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
                inputs = data['x'].to(device)
                conds = data['cond'].to(device)
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
                        inputs=data_batch['x'].to(device),
                        conds=data_batch['cond'].to(device),
                        point=point
                    )
                )
        return torch.cat(log_prob, dim=0).detach().numpy()

    