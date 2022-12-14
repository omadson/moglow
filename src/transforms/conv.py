import scipy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nflows import transforms

from src.transforms import thops


def nan_throw(tensor, name="tensor"):
        stop = False
        if ((tensor!=tensor).any()):
            print(name + " has nans")
            stop = True
        if (torch.isinf(tensor).any()):
            print(name + " has infs")
            stop = True
        if stop:
            print(name + ": " + str(tensor))
            #raise ValueError(name + ' contains nans of infs')

class InvertibleConv1x1(transforms.Transform):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            #self.p = torch.Tensor(np_p.astype(np.float32))
            #self.sign_s = torch.Tensor(np_sign_s.astype(np.float32))
            self.register_buffer('p', torch.Tensor(np_p).double())
            self.register_buffer('sign_s', torch.Tensor(np_sign_s).double())
            self.l = nn.Parameter(torch.Tensor(np_l).double())
            self.log_s = nn.Parameter(torch.Tensor(np_log_s).double())
            self.u = nn.Parameter(torch.Tensor(np_u).double())
            self.l_mask = torch.Tensor(l_mask).double()
            self.eye = torch.Tensor(eye).double()
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, inputs, reverse):
        w_shape = self.w_shape
        if not self.LU:
            timesteps = int(inputs.size(2))
            dlogdet = torch.slogdet(self.weight)[1] * timesteps
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1)
            else:
                weight = torch.inverse(self.weight.double()).double()\
                              .view(w_shape[0], w_shape[1], 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(inputs.device)
            self.sign_s = self.sign_s.to(inputs.device)
            self.l_mask = self.l_mask.to(inputs.device)
            self.eye = self.eye.to(inputs.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * int(inputs.size(2))
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).double()
                u = torch.inverse(u.double()).double()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1), dlogdet

    def forward(self, inputs, context=None):
        """
        log-det = log|abs(|W|)| * timesteps
        """
        weight, dlogdet = self.get_weight(inputs, reverse=False)
        nan_throw(weight, "weight")
        nan_throw(dlogdet, "dlogdet")
        z = F.conv1d(inputs, weight)
        return z, dlogdet

    def inverse(self, inputs, context=None):
        nan_throw(inputs, "InConv inpust")
        weight, dlogdet = self.get_weight(inputs, reverse=True)
        z = F.conv1d(inputs, weight)
        nan_throw(z, "InConv z")
        return z, dlogdet