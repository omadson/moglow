import torch
from torch import nn
from nflows import transforms

class CompositeTransform(transforms.Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, conds, funcs, context, point):
        batch_size = inputs.shape[0]
        outputs = inputs
        if point:
            total_logabsdet = torch.zeros(batch_size, inputs.shape[2])
        else:
            total_logabsdet = inputs.new_zeros(batch_size)
        for func in funcs:
            if point:
                
                outputs, _, logabsdet = func(outputs, conds, context, point)
            else:
                outputs, logabsdet = func(outputs, conds, context)           
            total_logabsdet += logabsdet
        if point:
            return outputs, total_logabsdet, total_logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, conds, context=None, point=False):
        funcs = self._transforms
        return self._cascade(inputs, conds, funcs, context, point)

    def inverse(self, inputs, conds, context=None, point=False):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, conds, funcs, context, point)