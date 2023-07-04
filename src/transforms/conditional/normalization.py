import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nflows.transforms.base import InverseNotAvailable, Transform
import nflows.utils.typechecks as check
from nflows.utils import torchutils


class ActNorm(Transform):
    def __init__(self, num_features, scale=1.0):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.

        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not check.is_positive_int(num_features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()
        # register mean and scale
        size = [1, num_features]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)

        with torch.no_grad():
            bias = inputs.mean(dim=0) * -1.0
            vars = ((inputs + bias) ** 2).mean()
            logs = torch.log(self.scale / (torch.sqrt(vars)+1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.initialized.data = torch.tensor(True, dtype=torch.bool)

    def _center(self, inputs, reverse=False):
        if reverse:
            return inputs - self.bias
        return inputs + self.bias
    
    def _scale(self, inputs, reverse=False, point=False):
        logs = self.logs
        multiplier = -1.0 if reverse else 1.0
        inputs = inputs * torch.exp(logs * multiplier)
        dlogdet = torchutils.sum_except_batch(logs, num_batch_dims=1) * multiplier
        if point:
            return inputs, dlogdet, logs
        return inputs, dlogdet
    
    def forward(self, inputs, conds=None, context=None, point=False):
        if self.training and not self.initialized:
            self._initialize(inputs)
        inputs = self._center(inputs, reverse=False)
        if point:
            inputs, logdet, point_log = self._scale(inputs, reverse=False, point=point)
            return inputs, logdet, point_log.squeeze().repeat(inputs.shape[0], 1)
        inputs, logdet = self._scale(inputs, reverse=False)
        return inputs, logdet * torch.ones(inputs.shape[0], device=inputs.device)

    def inverse(self, inputs, conds=None, context=None):
        # scale and center
        inputs, logdet = self._scale(inputs, reverse=True)
        inputs = self._center(inputs, reverse=True)
        return inputs, logdet

    
    
    


def nan_throw(tensor, name="tensor"):
    stop = False
    if ((tensor!=tensor).any()):
        print(name + " has nans @ normalization class")
        stop = True
    if (torch.isinf(tensor).any()):
        print(name + " has infs @ normalization class")
        stop = True
    if stop:
        print(name + ": " + str(tensor))
        #raise ValueError(name + ' contains nans of infs')