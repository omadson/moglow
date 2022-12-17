import torch
from torch import optim
from torch.utils.data import DataLoader
from nflows import transforms, flows, distributions

from .flow import Flow

from .transforms import (
    CompositeTransform,
    ActNorm2d,
    InvertibleConv1x1,
    AffineCouplingTransform,
    ReshapeTransform
)



def create_transform_step(num_channels, cond_channels, level_hidden_channels, network):
    return CompositeTransform([
        # 1. actnorm
        ActNorm2d(num_channels), 
        # 2. permute
        InvertibleConv1x1(num_channels, LU_decomposed=True), 
        # 3. coupling
        AffineCouplingTransform(
            num_channels,
            cond_channels,
            level_hidden_channels,
            network=network
        ) 
    ])

def create_transform(num_channels, cond_channels, seq_len, levels, hidden_channels, network):
    hidden_channels = [hidden_channels] * levels
    all_transforms = []
    for level, level_hidden_channels in zip(range(levels), hidden_channels):
        all_transforms.append(
            create_transform_step(
                num_channels,
                cond_channels,
                level_hidden_channels,
                network
            )
        )
    all_transforms.append(ReshapeTransform(
        input_shape = (num_channels, seq_len),
        output_shape = (num_channels * seq_len,)
    ))
    return CompositeTransform(all_transforms)


def create_flow(num_channels, cond_channels, seq_len, levels, hidden_channels, network, device=None):
    distribution = distributions.StandardNormal((num_channels * seq_len,)).double().to(device)
    transform = create_transform(
        num_channels=num_channels,
        cond_channels=cond_channels,
        seq_len=seq_len,
        levels=levels,
        hidden_channels=hidden_channels,
        network=network
    )
    flow = Flow(transform, distribution).double()
    return flow


class Moglow:
    def __init__(self, num_channels, cond_channels, seq_len, levels=3, hidden_channels=128, network='LSTM', device=None):
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.flow = create_flow(
            num_channels=num_channels,
            cond_channels=cond_channels,
            seq_len=seq_len,
            levels=levels,
            hidden_channels=hidden_channels,
            network=network,
            device=self.device
        )

    def train(self, train_set, batch_size=128, max_epochs=50, log_interval=10, learning_rate=1e-4, weight_decay=1e-2):
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.flow.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_list = []
        for epoch in range(1, max_epochs+1):
            loss_all = torch.tensor(0, dtype=torch.float, device=self.device)
            for data_batch in train_loader:
                self.flow.train()
                optimizer.zero_grad()
                loss = -self.flow.log_prob(inputs=data_batch['x'], conds=data_batch['cond']).mean()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_all += loss.sum()
            loss_list.append(loss_all.item()/len(train_loader.dataset))
            if epoch % log_interval == 0:
                print(f" - Epoch {epoch:3d}: {loss.item():.3f}")
        return loss_list
        
    def sample(self, num_samples=100, conds=None):
        return self.flow.sample_and_log_prob(num_samples, conds)

