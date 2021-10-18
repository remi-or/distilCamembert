from typing import Optional, List, Union
import torch
import matplotlib.pyplot as plt
from utils.metrics import Metric
from time import perf_counter
from datetime import datetime

Tensor = torch.Tensor


class Logger:

    def __init__(
        self,
        metrics : List[Union[Metric, str]],
        savepath : Optional[str] = None,
        ) -> None:
        self.savepath = savepath + datetime.now().strftime("%d.%m.%Y-%H:%M:%S") + '-logs.json' if savepath is not None else None
        self.phase = ''
        self.metrics = {}
        self.t0 = None
        self.times = []
        for m in metrics:
            if isinstance(m, str):
                self.metrics[m] = Metric.from_description(m)
            else:
                self.metrics[m.active_phase + ' ' + m.nature] = m

    def __len__(self, ) -> int:
        return max(len(m) for m in self.metrics)

    def __getitem__(
        self,
        description : str,
        ) -> Metric:
        return self.metrics[description]

    def train(self, ) -> None:
        self.t0 = perf_counter()
        for _, m in self.metrics.items():
            m.train()
        self.phase = 'train'

    def val(self, ) -> None:
        for _, m in self.metrics.items():
            m.val()
        self.phase = 'val'

    def save_to_json(self,) -> None:
        if self.savepath is not None:
            with open(self.savepath, mode='a') as file:
                file.write(self.phase)
                file.write('\n')
                for name, metric in self.metrics.items():
                    if metric.phase == self.phase:
                        file.write(name)
                        file.write(': ')
                        file.write(str(metric[-1]))
                        file.write('\n')
                file.write('\n')

    def log_batch(
        self,
        problem_type : str,
        loss : float,
        forward_outputs : Tensor,
        labels : Tensor,
        ) -> None:
        for _, m in self.metrics.items():
            m.log_batch(problem_type, loss, forward_outputs, labels)

    def end(
        self, 
        ) -> None:
        for _, m in self.metrics.items():
            m.end()
        self.save_to_json()
        if self.phase == 'val' and self.t0 is not None:
            self.times.append(perf_counter() - self.t0)
            self.t0 = None
        self.phase = ''

    def state_dict(self, ) -> dict:
        sd = {}
        sd['phase'] = self.phase
        sd['metrics'] = self.metrics
        sd['times'] = self.times
        sd['savepath'] = self.savepath
        return sd

    def load_state_dict(self, sd : dict) -> None:
        self.metrics = sd['metrics']
        self.phase = sd['phase']
        self.times = sd['times']
        self.savepath = sd['savepath']

    def plot_loss(self,) -> None:
        self.plot_metric('loss')

    def plot_accuracy(self, ) -> None:
        self.plot_metric('accuracy')

    def plot_metric(
        self,
        nature : str,
        ) -> None:
        for phase in ['train', 'val']:
            description = phase + ' ' + nature
            if description in self.metrics:
                y = self.metrics[description].history
                x = [i for i in range(len(y))]
                plt.plot(x, y, label=description)
        plt.legend()
        plt.show()