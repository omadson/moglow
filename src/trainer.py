from tqdm.notebook import trange, tqdm
from torch.utils.data import DataLoader
from torch import optim


class Trainer:
    def __init__(
            self,
            max_epochs=100,
            batch_size=64,
            weight_decay=1e-5,
            learning_rate=1e-2,
            loss_function=None,
            max_time=-1
        ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_time = max_time # TODO: stop training cycle with a max time
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.loss_function = loss_function
    
    def loss_func(self, model, data_batch):
        if not self.loss_function:
            return -model.log_prob(
                inputs=data_batch['x'],
                conds=data_batch['cond']
            ).mean()
        return self.loss_function(model, data_batch)

    def fit(self, model, train_set, log_times=10):
        train_dataloader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        log_interval = int(self.max_epochs / log_times)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.training_loss = []
        for epoch in trange(self.max_epochs, desc="Training model", unit="epoch"):
            running_loss = 0.0
            model.train()
            for i, data_batch in enumerate(train_dataloader):
                if epoch == 0:
                    model.init_lstm_hidden(data_batch['x'].shape[0])
                model.repackage_lstm_hidden()
                loss = self.loss_func(model, data_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()
            epoch_loss = running_loss / (i+1)
            self.training_loss.append(epoch_loss)
            if (epoch % log_interval == 0) or (epoch == 0):
                print(f" - Epoch {epoch:3d}/{self.max_epochs:3d}: {epoch_loss:.3f}")
