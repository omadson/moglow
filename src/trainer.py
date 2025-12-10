from pathlib import Path
from tqdm.notebook import trange, tqdm
from torch.utils.data import DataLoader
from torch import optim
import torch
import numpy as np


class Trainer:
    def __init__(
            self,
            dataset,
            model,
            loss_function=None,
            max_epochs=100,
            batch_size=64,
            weight_decay=1e-5,
            learning_rate=1e-2,
            log_times=10,
            device: torch.device = torch.device("cpu")
        ):
        self.dataset = dataset
        self.model = model
        self.loss_function = loss_function
        # optimizer and scheduler
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.log_times = log_times
        self.device = device
        # prepare data loaders
        self._prepare_loaders()
        # create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        # create scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)

    def _prepare_loaders(self):
        ...
        
    def _train_loop(self):
        running_loss = []
        self.model.train()
        for i, data in enumerate(self.train_loader):
            inputs = data['x'].to(self.device).squeeze(dim=2)
            conds = data['cond'].to(self.device).squeeze(dim=1)
            if i == 0: self.model.prepare_lstm()
            else: self.model.prepare_lstm(repackage=True)
            self.optimizer.zero_grad()
            loss = self.loss_function(self.model, inputs, conds)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.cpu().item())
        self.scheduler.step()
        return np.average(running_loss)
     
    def _valid_loop(self):
        ...
    
    def train(self):
        ...