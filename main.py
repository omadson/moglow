from typing import Optional
from enum import Enum
from pathlib import Path
import shutil

import yaml
import typer
from typer import Option, Argument
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper.stopper import Stopper
from rich.console import Console
from rich.table import Table

from src.models.moglow import Moglow, MoglowConfig, MoglowTrainer
from src.models.tcnf import TCNFTrainer
from src.data.utils import load_data
from src.metrics.pot import pot_eval
from src.metrics.diagnosis import hit_att, ndcg


trainers = {
    'TCNF': TCNFTrainer,
    'Moglow': MoglowTrainer
}

def get_metric_results(train_loss, test_loss, labels, dataset):
    df = pd.DataFrame()
    for i in range(test_loss.shape[1]):
        lt, l, ls = train_loss[:, i], test_loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls, dataset=dataset); #preds.append(pred)
        df = df.append(result, ignore_index=True)
    final_train_loss, final_test_loss = np.mean(train_loss, axis=1), np.mean(test_loss, axis=1)
    final_labels = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(final_train_loss, final_test_loss, final_labels, dataset=dataset)
    result.update(hit_att(test_loss, labels))
    result.update(ndcg(test_loss, labels))
    return result

# class EarlyStopper(Stopper):
#     def __init__(self, patience=5, min_delta=0.0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = {}
#         self.best = {}
    
#     def __call__(self, trial_id, result):
#         valid_loss = result["valid_loss"]
#         if not self.best.get(trial_id):
#             self.best[trial_id] = valid_loss
#             self.counter[trial_id] = 0
#         elif valid_loss > self.best[trial_id] + self.min_delta:
#             self.counter[trial_id] += 1
#             if self.counter[trial_id] > self.patience:
#                 return True
#         else:
#             self.best[trial_id] = valid_loss
#             self.counter[trial_id] = 0
#         return False
    
#     def stop_all(self):
#         return False


class TrainModel(tune.Trainable):
    def setup(self, config):
        tune_params = config['tune_params']
        train_params = config['train_params']
        use_cuda = tune_params['use_gpu'] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.datasets = load_data(
            name=train_params['dataset'],
            sequence_length=train_params['length'],
        )
        self.train_loader = DataLoader(
            self.datasets['window_train'], 
            batch_size=train_params['batch_size'],
            shuffle=True
        )
        self.valid_loader = DataLoader(
            self.datasets['window_valid'], 
            batch_size=train_params['batch_size'],
            shuffle=True
        )

        self.learning_rate = train_params['learning_rate']
        self.weight_decay = train_params['weight_decay']

        self.trainer = trainers[train_params['model']]

        self.model = self.trainer.create(
            self.datasets['window_train'].info,
            config,
            self.device
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.best_valid_score = None

    def step(self):
        # train step
        train_loss = self.trainer.train(self.model, self.optimizer, self.scheduler, self.train_loader, device=self.device)
        # valid step
        valid_loss = self.trainer.validation(self.model, self.valid_loader, device=self.device)
        if (self.best_valid_score == None) or (valid_loss < self.best_valid_score):
            self.best_valid_score = valid_loss
        return {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "best_valid_loss": self.best_valid_score,
        }

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = f"{checkpoint_dir}/model.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class Datasets(Enum):
    mba = 'MBA'
    nab = 'NAB'
    msl = 'MSL'
    smap = 'SMAP'
    smd = 'SMD'
    swat = 'SWaT'
    ucr = 'UCR'
    wadi = 'WADI'
    synthetic = 'synthetic'


class Models(Enum):
    moglow = 'Moglow'
    tcnf = 'TCNF'


def conf_callback(ctx: typer.Context, param: typer.CallbackParam, value: str):
    if value:
        try: 
            with open(value, 'r') as f:    # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}   # Initialize the default map
            ctx.default_map.update(conf)   # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return value

def folder_name(model_name, dataset_name, model_params):
    net = 'lstm' if model_params['recurrent_network'] else 'ff'
    return f'{model_name}_{net}_{dataset_name}'

app = typer.Typer()

@app.command("train")
def train_model(
        model: Models = Argument(Models.moglow.value, help="Model name"),
        dataset: Datasets = Argument(Datasets.mba.value, help="Dataset name"),
        length: int = Argument(2, help="Sequence length of the dataset"),
        # model config
        model_config: Path = Option("model.yaml", help="Configuration file for train options"),
        # Train options
        train_config: Optional[Path] = Option('config/train.yaml', callback=conf_callback, is_eager=False, help="Configuration file for train options"),
        epochs: int = Option(500, min=1, max=1000, help="Number of epochs of the training"),
        batch_size: int = Option(64, min=1, max=512, help="Number of samples per batch"),
        weight_decay: float = Option(1e-2, min=1e-6, max=1, help="Weight decay coefficient"),
        learning_rate: float = Option(1e-3, min=1e-6, max=1, help="Learning rate"),
        # Tune options
        tune_config: Optional[Path] = Option('config/tune.yaml', callback=conf_callback, is_eager=False, help="Configuration file for tune options"),
        trials: int = Option(50, min=1, max=200, help="Number of trials into grid search"),
        use_gpu: bool = Option(False, help="Enable CUDA training"),
        num_cpus: int = Option(1, min=1, max=8, help="Number of CPUs used into grid search"),
        results_folder: str = Option('./results', help="Local dir to save training results to"),
        replace: bool = Option(False, help="Replace existing results and start a new grid search")    
    ):
    model_params = {}
    if model_config.exists():
        with open(model_config, 'r') as f: 
            model_params = yaml.safe_load(f)
    
    train_params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'learning_rate': learning_rate,
        'dataset': dataset.value,
        'length': length,
        'model': model.value
    }

    tune_params = {
        'trials': trials,
        'use_gpu': use_gpu,
        'num_cpus': num_cpus,
        'results_folder': results_folder,
    }
    results_name = folder_name(model.value, dataset.value, model_params)
    experiment_path = Path(f"{results_folder}/{results_name}")
    if replace and experiment_path.exists():
        shutil.rmtree(experiment_path)
    if not experiment_path.exists():
        typer.echo("Starting a new grid search experiment.")
        ray.init(num_cpus=num_cpus)
        sched = ASHAScheduler(
            metric="valid_loss",
            mode="min",
            max_t=epochs,
            grace_period=int(epochs*.01)+1,
            reduction_factor=2
        )
        tuner = tune.Tuner(
            tune.with_resources(
                TrainModel,
                resources={
                    "cpu": num_cpus,
                    "gpu": 1 if tune_params['use_gpu'] else 0
                }
            ),
            tune_config=tune.TuneConfig(
                # metric="loss",
                # mode="min",
                scheduler=sched,
                num_samples=tune_params['trials']
            ),
            param_space={
                'tune_params': tune_params,
                'train_params': train_params,
                'weight_decay': train_params['weight_decay'],
                'learning_rate': train_params['learning_rate'],
                **{key: tune.choice(value) if isinstance(value, list) else value for key, value in model_params.items()}
            },
            run_config=ray.air.config.RunConfig(
                local_dir=results_folder,
                name=results_name,
                checkpoint_config=ray.air.config.CheckpointConfig(
                    checkpoint_score_attribute="valid_loss",
                    checkpoint_score_order="min",
                    num_to_keep=1,
                    checkpoint_frequency=1,
                    checkpoint_at_end=False
                ),
                failure_config=air.FailureConfig(fail_fast=True),
                # stop=EarlyStopper(patience=15)
            )
        )
        results = tuner.fit()
    else:
        typer.echo("Restoring an old grid search experiment.")
        tuner = tune.Tuner.restore(
            path=str(experiment_path)
        )
        results = tuner.get_results()
    best_trial = results.get_best_result("best_valid_loss", "min")

    datasets = load_data(
        name=dataset.value,
        sequence_length=length,
    )
    use_cuda = tune_params['use_gpu'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    trainer = trainers[train_params['model']]

    best_trained_model = trainer.create(datasets['window_train'].info, best_trial.config, device)

    # test best model
    best_checkpoint = torch.load(Path(best_trial.checkpoint.to_directory()) / "model.pth")
    best_trained_model.load_state_dict(best_checkpoint)

    train_set = torch.utils.data.ConcatDataset([datasets['window_train'], datasets['window_valid']])
    # train_score = trainer.get_scores(best_trained_model, train_set, device).cpu().detach().numpy()

    test_loss = trainer.get_scores(best_trained_model, datasets['window_test'], device, point=True)#.cpu().detach().numpy()
    train_loss = trainer.get_scores(best_trained_model, train_set, device, point=True)#.cpu().detach().numpy()
    labels = datasets['labels']

    metrics = get_metric_results(train_loss, test_loss, labels, dataset=dataset.value)

    metrics.update(best_trial.config)
    metrics['name'] = dataset.value
    metrics['best_valid_loss'] = best_trial.metrics['best_valid_loss']
    metrics['last_valid_loss'] = best_trial.metrics['valid_loss']
    metrics['last_train_loss'] = best_trial.metrics['train_loss']

    table = Table(title="Best model metrics.")

    table.add_column("Param", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", justify="left", style="green")

    keys = ['name', 'precision', 'recall', 'ROC/AUC', 'f1', 'Hit@100%', 'Hit@100%', 'NDCG@100%', 'NDCG@150%']

    keys.extend([key for key in metrics.keys() if key not in keys])
    final_metrics = {key.lower(): metrics.get(key) for key in keys}

    for key, value in final_metrics.items():
        if key not in ['tp', 'fp', 'tn', 'fn', 'threshold', 'tune_params', 'train_params']:
            if key in ['weight_decay']:
                table.add_section()
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, f"{value}")
        

    console = Console()
    console.print(table)

@app.command("preprocess")
def preprocessing_data(
        dataset: Datasets = Argument(Datasets.mba.value, help="Dataset name"),
        length: int = Argument(10, help="Sequence length of the dataset"),
    ):
    ...


if __name__ == "__main__":
    app()