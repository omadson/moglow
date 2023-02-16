from sys import getsizeof
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from nflows import transforms, distributions

from .flow import Flow
from .distribuitions import StandardNormal
from .transforms import (
    thops,
    CompositeTransform,
    ActNorm2d,
    InvertibleConv1x1,
    AffineCouplingTransform,
    ReshapeTransform
)


class Moglow(Flow):
    def __init__(
        self,
        features,
        conditional_features,
        sequence_length,
        num_layers=3,
        coupling_flow='affine',
        coupling_network='LSTM',
        hidden_features=128,
        num_blocks_per_layer=2
    ):
        self.num_blocks_per_layer = num_blocks_per_layer
        self.hidden_features = hidden_features

        layers = []
        for _ in range(num_layers):
            layers.append(CompositeTransform([
                # 1. actnorm
                ActNorm2d(features), 
                # 2. permute
                InvertibleConv1x1(features, LU_decomposed=True), 
                # 3. coupling
                AffineCouplingTransform(
                    in_channels=features,
                    cond_channels=conditional_features,
                    hidden_channels=hidden_features,
                    network=coupling_network,
                    num_blocks_per_layer=num_blocks_per_layer,
                    flow_coupling=coupling_flow
                ) 
            ]))
        
        # layers.append(ReshapeTransform(
        #     input_shape = (features, sequence_length),
        #     output_shape = (features * sequence_length,)
        # ))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal((features, sequence_length,)),
        )
        
    def init_lstm_hidden(self, batch_size):
        # for step_transforms in self._transform.children():
        #     for step_transform in step_transforms:
        #         for transform in step_transform.children():
        #             for inner_transform in transform:
        #                 if isinstance(inner_transform, AffineCouplingTransform):
        #                     if inner_transform.network.lower() == 'lstm':
        #                         inner_transform.f.init_hidden(inputs['x'].shape[0])
        #                         z1, z2 = thops.split_feature(inputs['x'], "split")
        #                         z1_cond = torch.cat((z1, inputs['cond']), dim=1)  
        #                         inner_transform.f(z1_cond.permute(0, 2, 1))
        layer_transforms = [layer._transforms[-1] for layer in next(self._transform.children())]
        for layer_transform in layer_transforms:
            if layer_transform.network.lower() == 'lstm':
                layer_transform.f.init_hidden(batch_size)
                
    
    def repackage_lstm_hidden(self):
        layer_transforms = [layer._transforms[-1] for layer in next(self._transform.children())]
        for layer_transform in layer_transforms:
            if layer_transform.network.lower() == 'lstm':
                layer_transform.f.hidden = tuple(
                    Variable(v.data) for v in layer_transform.f.hidden
                )
        
         # for step_transforms in self._transform.children():
         #    for step_transform in step_transforms:
         #        for transform in step_transform.children():
         #            for inner_transform in transform:
         #                if isinstance(inner_transform, AffineCouplingTransform):
         #                    if inner_transform.network.lower() == 'lstm':
         #                        inner_transform.f.hidden = tuple(
         #                            Variable(v.data) for v in inner_transform.f.hidden
         #                        )
                            
    def concat_sequence(self, seqlen, data):
        """ 
        Concatenates a sequence of features to one.
        """
        nn,n_timesteps,n_feats = data.shape
        L = n_timesteps-(seqlen-1)
        inds = np.zeros((L, seqlen)).astype(int)

        #create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        for ii in range(0,seqlen):  
            inds[:, ii] = rng[ii:(n_timesteps-(seqlen-ii-1))]

        # slice each sample into L sequences and store as new samples 
        cc = data[:,inds,:].copy()

        #print ("cc: " + str(cc.shape))

        #reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen*n_feats))
        #print ("dd: " + str(dd.shape))
        return dd
    
    def generate(self, n_samples, autoreg, sequence_length):
        autoreg_all = autoreg
        autoreg = np.zeros_like(autoreg_all) # initialize from a mean pose
        autoreg[:,:,0] = autoreg_all[:,:,0]
        ntravels, nfeats, nsteps = autoreg_all.shape
        nfeats = int(nfeats / sequence_length)
        sampled_all = np.zeros((ntravels, nfeats, nsteps+sequence_length-1))
        sampled_all[:,:,:sequence_length] = (
            autoreg_all[:,:,0]
            .reshape(-1, sequence_length, nfeats)
            .permute(0, 2, 1)
        )
        # Loop through control sequence and generate new data
        for i in range(0, autoreg_all.shape[2]-sequence_length+1):
            # sample from Moglow
            self.eval()
            if i == 0:
                self.init_lstm_hidden(autoreg.shape[0])
            else:
                # self.repackage_lstm_hidden()
                self.init_lstm_hidden(autoreg.shape[0])
            sampled = self.sample(n_samples, conds=autoreg)
            
            # store the sampled frame
            sampled_all[:,:,sequence_length:(i+sequence_length)] = sampled.detach().numpy()[:,:,:i]
            autoreg = (
                self
                .concat_sequence(
                    sequence_length,
                    sampled_all.swapaxes(1,2)
                ).swapaxes(1,2)
            )
        return sampled_all

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)