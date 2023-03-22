import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List, Callable

from src.flow import Flow


class GRADINGS:
    def __init__(self, model: Flow, aggregation_functions: List[Callable], batch_size: int=64):
        self.model = model
        self.aggregation_functions = aggregation_functions
        self.batch_size = batch_size

    def predict(self, dataset: DataLoader):
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
        )
        self.model.eval()
        with torch.no_grad():
            log_prob = []
            for data_batch in data_loader:
                log_prob.append(
                    -self
                    .model
                    .log_prob(
                        inputs=data_batch['x'],
                        conds=data_batch['cond']
                    )
                )
        result = pd.DataFrame({
            'serie': dataset[:]['serie'],
            'label': dataset[:]['label'],
            'score': torch.cat(log_prob, dim=0)
        })
        scores = (
            result
            .groupby(['serie', 'label'])
            .agg(self.aggregation_functions)
            .reset_index()
        )
        scores.columns = scores.columns.map(lambda x: x[0] if not x[1] else f"{x[1]}_{x[0]}")
        return scores