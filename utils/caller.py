from typing import Union, Optional
from utils.logger import Logger
from utils.metrics import Metric
from torch.optim import Optimizer
import torch


Module = torch.nn.Module # Not our custom class, but avoids a circular import


class Caller:

    def __init__(
        self,
        savepath : Optional[str] = None,
    ) -> None:
        self.early_stopping = {'on' : False}
        self.checkpoint = {'on' : False}
        self.savepath = savepath

    def add_checkpoint(
        self,
        frequency : Optional[int],
        save_opt : bool = False,
        save_logger : bool = False,
    ) -> None:
        if self.savepath is None:
            raise(ValueError("Please enter a savepath first."))
        self.checkpoint['on'] = True
        self.checkpoint['frequency'] = frequency
        self.checkpoint['save_opt'] = save_opt
        self.checkpoint['save_logger'] = save_logger

    def _checkpoint(
        self,
        model : Module,
        optimizer : Optimizer,
        logger : Logger,
        i_epoch : int,
        is_stopping : bool,
    ) -> None:
        if not self.checkpoint['on']:
            return None
        freq = self.checkpoint['frequency']
        if freq is None:
            save = is_stopping
        else:
            save = (i_epoch + 1) % freq == 0
        if save:
            optimizer = optimizer if self.checkpoint['save_opt'] else None
            logger = logger if self.checkpoint['save_logger'] else None
            self.save(model, optimizer, logger, self.savepath + f"checkpoint_{len(logger)}_")

    def save(
        self,
        model : Module,
        optimizer : Optional[Optimizer] = None,
        logger : Optional[Logger] = None,
        savepath : Optional[str] = None,
    ) -> None:
        savepath = self.savepath if savepath is None else savepath
        torch.save(model.state_dict(), savepath + 'model.pt')
        if optimizer is not None:
            torch.save(optimizer.state_dict(), savepath + 'optimizer.pt')
        if logger is not None:
            torch.save(logger.state_dict(), savepath + 'logger.pt')       

    def load(
        self,
        model : Module,
        optimizer : Optional[Optimizer] = None,
        logger : Optional[Logger] = None,
        savepath : Optional[str] = None,
    ) -> None:
        savepath = self.savepath if savepath is None else savepath
        model.load_state_dict(torch.load(savepath + 'model.pt'))
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(savepath + 'optimizer.pt'))
        if logger is not None:
            logger.load_state_dict(torch.load(savepath + 'logger.pt'))

    def add_early_stopping(
        self,
        patience : int,
        metric : Union[Metric, str],
        warmup : int = 0,
    ) -> None:
        self.early_stopping['on'] = True
        self.early_stopping['patience'] = patience
        self.early_stopping['metric'] = metric if isinstance(metric, Metric) else Metric.from_description(metric)
        self.early_stopping['waited'] = patience + warmup
        self.early_stopping['best'] = None

    def _early_stopping(
        self,
        logger : Logger,
    ) -> bool:
        # Check if the callback is on
        if not self.early_stopping['on']:
            return False
        # Retrieve the metric
        metric = self.early_stopping['metric']
        # Find the last measure
        last = logger[metric.phase + ' ' + metric.nature][-1]
        if last is None:
            raise(ValueError(f'Metric {metric} not found during ealry stopping callback.'))
        # Retrieve best
        best = self.early_stopping['best']
        if best is None:
            self.early_stopping['best'] = last
        # Compare
        else:
            if metric.best(best, last) != best:
                self.early_stopping['best'] = last
                self.early_stopping['waited'] = max(self.early_stopping['patience'], self.early_stopping['waited'])
            else:
                self.early_stopping['waited'] -=1
        return self.early_stopping['waited'] == 0

    def __call__(
        self,
        model : Module,
        optimizer : Optimizer,
        logger : Logger,
        i_epoch : int,
        epochs : int,
    ) -> bool:
        stop = (i_epoch+1 == epochs)
        self._checkpoint(model, optimizer, logger, i_epoch, stop)
        stopping_early = self._early_stopping(logger)
        return stopping_early