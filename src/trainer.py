from pathlib import Path
from tqdm.notebook import trange, tqdm
from torch.utils.data import DataLoader
from torch import optim
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.early_stop = False

    def __call__(self, epoch_loss):
        if epoch_loss['valid'][0]:
            train_loss = np.array(epoch_loss['train'])
            valid_loss = np.array(epoch_loss['valid'])
            counter = np.sum((valid_loss - train_loss) > self.min_delta)
            if counter >= self.tolerance: 
                return True
        return False

class Trainer:
    def __init__(
            self,
            max_epochs=100,
            batch_size=64,
            weight_decay=1e-5,
            learning_rate=1e-2,
            loss_function=None,
            log_times=10,
            dataset_name: str = None,
        ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.log_interval = int(self.max_epochs / log_times)
        self.dataset_name = dataset_name
    
    def _loss_func(self, model, data_batch, epoch):
        if not self.loss_function:
            return -model.log_prob(
                inputs=data_batch['x'],
                conds=data_batch['cond']
            ).mean()
        return self.loss_function(model, data_batch, epoch)

    def _lstm_step(self, epoch, model, batch):
        if epoch == 0:
            model.init_lstm_hidden(batch['x'].shape[0])
        model.repackage_lstm_hidden()

    def fit(self, model, train_set, valid_set=None):
        # prepare data loaders
        train_loader, valid_loader = self._prepare_loaders(train_set, valid_set)
        # load checkpoint
        model, checkpoint_epoch = self._load_checkpoint(model)
        self._early_stopping = EarlyStopping(tolerance=5, min_delta=0)
        for epoch in trange(1, self.max_epochs+1):
            if epoch <= checkpoint_epoch: continue
            self.epoch_loss['train'].append(
                self._train_loop(model, train_loader, epoch), # model train
            )
            self.epoch_loss['valid'].append(
                self._valid_loop(model, valid_loader, epoch) # model validation
            )
            # early stop
            if self._early_stopping(self.epoch_loss):
                print(" - Stopping in epoch :", epoch)
                break
            self._save_checkpoint(model, epoch)
            # logging
            if (epoch % self.log_interval == 0) or (epoch == 1):
                print(
                    f" - Epoch {epoch:3d}/{self.max_epochs:3d},",
                    f"train_loss={self.epoch_loss['train'][-1]:.3f},",
                    f"valid_loss={self.epoch_loss['valid'][-1]:.3f}"
                )

    def _train_loop(self, model, train_loader, epoch):
        train_losses = []
        model.train()
        for batch in train_loader:
            loss = self._loss_func(model, batch, epoch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
        self.scheduler.step()
        return np.average(train_losses)
    
    def _valid_loop(self, model, valid_loader, epoch):
        if valid_loader: 
            valid_losses = []
            model.eval()
            with torch.no_grad():
                for batch in valid_loader:
                    loss_valid = self._loss_func(model, batch, epoch)
                    valid_losses.append(loss_valid.item())
            return np.average(valid_losses)
        else:
            return None
    
    def _prepare_loaders(self, train_set, val_set):
        train_dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True) if val_set else None
        return train_dataloader, val_dataloader

    def _load_checkpoint(self, model):
        folder = Path(f'checkpoints/{model.name}_{self.dataset_name}/')
        file_path = folder / 'model.ckpt'
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.epoch_loss = {'train': [], 'valid': []}
        if file_path.exists():
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch_loss = checkpoint['epoch_loss']
            return model, checkpoint['epoch']
        else:
            return model, 0
    
    def _save_checkpoint(self, model, epoch):
        folder = Path(f'checkpoints/{model.name}_{self.dataset_name}/')
        folder.mkdir(exist_ok=True)
        file_path = folder / 'model.ckpt'
        torch.save({
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch_loss': self.epoch_loss,
            'epoch': epoch,
        }, file_path)
