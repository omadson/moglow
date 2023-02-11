
import torch

from nflows import transforms


class ReshapeTransform(transforms.Transform):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs, conds=None, context=None, point=False):
        if tuple(inputs.shape[1:]) != self.input_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.input_shape))
        if point:
            return (
                inputs.reshape(-1, *self.output_shape),
                torch.zeros(1, device=inputs.device),
                torch.zeros(1, device=inputs.device)
            )
        return (
            inputs.reshape(-1, *self.output_shape),
            torch.zeros(1, device=inputs.device)
        )

    def inverse(self, inputs, conds=None, context=None):
        if tuple(inputs.shape[1:]) != self.output_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.output_shape))
        return inputs.reshape(-1, *self.input_shape), torch.zeros(inputs.shape[0], device=inputs.device)