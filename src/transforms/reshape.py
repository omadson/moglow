
import torch

from nflows import transforms


class ReshapeTransform(transforms.Transform):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.input_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.input_shape))
        return inputs.reshape(-1, *self.output_shape).double(), torch.zeros(inputs.shape[0], device=inputs.device).double()

    def inverse(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.output_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.output_shape))
        return inputs.reshape(-1, *self.input_shape).double(), torch.zeros(inputs.shape[0], device=inputs.device).double()