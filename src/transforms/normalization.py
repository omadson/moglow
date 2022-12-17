from torch import nn
import torch
from nflows import transforms

from src.transforms import thops


class _ActNorm(transforms.Transform):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        assert input.device == self.bias.device
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=[0, 2], keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=[0, 2], keepdim=True)
            logs = torch.log(self.scale/(torch.sqrt(vars)+1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, reverse=False):
        logs = self.logs
        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)
        
        dlogdet = thops.sum(logs) * thops.timesteps(input)
        if reverse:
            dlogdet = -thops.sum(logs) * thops.timesteps(input)
        return input, dlogdet

    def forward(self, input, conds=None, context=None):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)
        # no need to permute dims as old version
        # center and scale
        input = self._center(input, reverse=False)
        input, logdet = self._scale(input, reverse=False)
        return input, logdet
    
    def inverse(self, input, conds=None, context=None):
        # scale and center
        input, logdet = self._scale(input, reverse=True)
        input = self._center(input, reverse=True)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 3
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCT`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))