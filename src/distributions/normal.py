"""Implementations of Normal distributions."""

import numpy as np
import torch
from torch import nn

from nflows.utils import torchutils

from ..flow import Distribution
from ..transforms import thops

class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)

        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _log_prob(self, inputs, conds, context, point=False):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -0.5 * torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
        if point:
            # point_neg_energy = -0.5 * (inputs ** 2)
            neg_point_energy = -0.5 * (inputs ** 2).sum(dim=2)
            print(neg_energy.shape)
            print(neg_point_energy.shape)
            return neg_energy - self._log_z, (neg_point_energy - (self._log_z / inputs.shape[1]))[:, :, None]
            # return neg_energy - self._log_z, neg_energy - self._log_z
        return neg_energy - self._log_z

    def _sample(self, num_samples, conds, context):
        if context is None:
            return torch.randn(num_samples, *self._shape, device=self._log_z.device)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape,
                                  device=context.device)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)