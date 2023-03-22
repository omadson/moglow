from pathlib import Path
from tqdm.notebook import trange, tqdm
from torch.utils.data import DataLoader
from torch import optim
import torch


class Trainer:
    def __init__(
            self,
            max_epochs=100,
            batch_size=64,
            weight_decay=1e-5,
            learning_rate=1e-2,
            loss_function=None,
            max_time=-1,
        ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_time = max_time # TODO: stop training cycle with a max time
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.loss_function = loss_function
    
    def loss_func(self, model, data_batch, epoch):
        if not self.loss_function:
            return -model.log_prob(
                inputs=data_batch['x'],
                conds=data_batch['cond']
            ).mean()
        return self.loss_function(model, data_batch, epoch)

    def _lstm_step(self, epoch, model, data_batch):
        if epoch == 0:
            model.init_lstm_hidden(data_batch['x'].shape[0])
        model.repackage_lstm_hidden()

    def fit(self, model, train_set, log_times=10):
        self.dataset_name = train_set.info()['name']
        train_dataloader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        log_interval = int(self.max_epochs / log_times)
        model, _epoch = self._prepare_model(model)
        for epoch in trange(_epoch+1, self.max_epochs, desc="Training model", unit="epoch"):
            running_loss = 0.0
            model.train()
            for i, data_batch in enumerate(train_dataloader):
                if model.name == "moglow":
                    self._lstm_step(epoch, model, data_batch)
                loss = self.loss_func(model, data_batch, epoch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()
            epoch_loss = running_loss / (i+1)
            self.training_loss.append(epoch_loss)
            self._save_model(
                model = model,
                optimizer = self.optimizer,
                scheduler = self.scheduler,
                epoch = epoch,
                training_loss = self.training_loss
            )
            if (epoch % log_interval == 0) or (epoch == 0):
                print(f" - Epoch {epoch:3d}/{self.max_epochs:3d}: {epoch_loss:.3f}")

    def _save_model(self, model, optimizer, scheduler, epoch, training_loss):
        folder = Path(f'checkpoints/{model.name}_{self.dataset_name}/')
        folder.mkdir(exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_loss': training_loss
        }, file_path)
    
    def _prepare_model(self, model):
        folder = Path(f'checkpoints/{model.name}_{self.dataset_name}/')
        file_path = Path(f'{folder}/model.ckpt')
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.training_loss = []
        if file_path.exists():
            print(f"Loading pre-trained model: {model.name}")
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            self.training_loss = checkpoint['training_loss']
        else:
            print(f"Creating new model: {model.name}")
            epoch = -1
        return model, epoch