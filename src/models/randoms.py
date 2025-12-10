from typing import Any
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from pydantic import PositiveInt, conint
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from ..distributions import StandardNormal
from ..flow import Flow
from ..transforms import conditional, CompositeTransform



class RANDOMS(Flow):
    def __init__(
            self,
            num_features: PositiveInt,
            num_conditional_features: PositiveInt,
            num_flows: conint(gt=0, lt=30) = 10,
            # coupling
            num_network_layers: conint(gt=0, lt=30) = 2,
            num_neurons_per_layer: conint(gt=2, lt=2**10) = 2**8,
            recurrent_network: bool = True,
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
                conditional.normalization.ActNorm(num_features),
                # 2. permute
                conditional.permutations.RandomPermutation(num_features),
                # 3. coupling
                conditional.coupling.AffineCouplingTransform(
                    num_features=num_features,
                    num_conditional_features=num_conditional_features,
                    num_network_layers=num_network_layers,
                    num_neurons_per_layer=num_neurons_per_layer,
                    recurrent_network=recurrent_network,
                )
            ])
        
        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal((num_features,)),
        )

    def prepare_lstm(self, repackage=False):
        for transform in next(self._transform.children()):
            if (
                transform._get_name() == 'AffineCouplingTransform' and
                self.recurrent_network
            ):
                if repackage:
                    transform.transform_net.hidden = tuple(
                        Variable(v.data) for v in transform.transform_net.hidden
                    )
                else:
                    transform.transform_net.init_hidden()

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class LitRANDOMS(L.LightningModule):
    def __init__(
            self,
            model,
            # loss_function=None,
            weight_decay=1e-5,
            learning_rate=1e-4,
    ):
        super().__init__()
        self.model = model
        # self.loss_function = loss_function
        # optimizer and scheduler
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
    
    def training_step(self, batch, batch_idx):
        inputs = batch['x'].squeeze(dim=2)
        conds = batch['cond']#.squeeze(dim=1)
        if batch_idx == 0: self.model.prepare_lstm()
        else: self.model.prepare_lstm(repackage=True)
        loss = -self.model.log_prob(inputs=inputs, conds=conds).mean()
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs = batch['x'].squeeze(dim=2)
        conds = batch['cond']#.squeeze(dim=1)
        if batch_idx == 0: self.model.prepare_lstm()
        else: self.model.prepare_lstm(repackage=True)
        loss = -self.model.log_prob(inputs=inputs, conds=conds).mean()
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    