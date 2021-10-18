from __future__ import annotations
from torch import Tensor
from typing import Union
import torch
import numpy as np


Array = np.ndarray


class Metric:

    @staticmethod
    def from_description(
        description : str,
        ) -> Metric:
        phase, nature = description.split(" ")
        if nature == 'loss':
            metric = Loss
        elif nature == 'accuracy':
            metric = Accuracy
        elif nature == 'map':
            metric = MAP
        elif nature.startswith('top') and nature.endswith('accuracy'):
            # If the metric is top k accuracy, the nature is in the form of top-k-accuracy
            k = int(nature.split('-')[1])
            metric = lambda phase : Top_k_accuracy(k, phase)
        return metric(phase)

    def __init__(
        self,
        phase : str,
        ) -> None:
        self.phase = phase
        self.running = False
        self.history = []
        self.acc_measures = []
        self.acc_sizes = 0

    def __len__(
        self,
        ) -> int:
        return len(self.history)

    def __getitem__(
        self,
        i : int,
        ) -> float:
        return self.history[i]

    def __repr__(
        self,
        ) -> str:
        return self.phase + ' ' + self.nature

    def reset_accumulators(
        self,
        ) -> None:
        self.acc_sizes = 0
        self.acc_measures = []

    def train(
        self,
        ) -> None:
        if self.phase == 'train':
            self.running = True
            self.reset_accumulators()

    def val(
        self,
        ) -> None:
        if self.phase == 'val':
            self.running = True
            self.reset_accumulators()

    def end(
        self,
        ) -> None:
        if not self.running:
            return None
        self.history.append(sum(self.acc_measures) / self.acc_sizes)
        self.running = False

    def log_batch(
        self,
        problem_type : str,
        loss : float,
        forward_outputs : Tensor,
        labels : Tensor,
        ) -> None:
        if not self.running:
            return None
        size = len(labels)
        self.acc_sizes += size
        measure = self.compute(problem_type, loss, forward_outputs, labels)
        self.acc_measures.append(measure * size)

    def state_dict(self, ) -> dict:
        sd = {}
        sd['phase'] = self.phase
        sd['history'] = self.history
        return sd

    def load_state_dict(self, sd : dict) -> None:
        self.history = sd['history']
        sd.phase = sd['phase']

class Loss(Metric):

    best = min
    nature = 'loss'

    @staticmethod
    def compute(
        problem_type : str,
        loss : float,
        forward_outputs : Tensor,
        labels : Tensor,
    ) -> float:
        return loss

class Accuracy(Metric):

    best = max
    nature = 'accuracy'

    @classmethod
    def compute(
        cls,
        problem_type : str,
        loss : float,
        forward_outputs : Tensor,
        labels : Tensor,
    ) -> float:
        if problem_type == 'binary_classification':
            return cls.binary_classification(forward_outputs, labels)
        elif problem_type == 'top_ranking':
            return cls.top_ranking(forward_outputs, labels)
        else:
            raise(ValueError(f"Unsupported problem type for accuracy metric: {problem_type}"))

    @staticmethod
    def binary_classification(
        forward_outputs : Tensor,
        labels : Tensor,
    ) -> float:
        good, batch_size = 0, 0
        for prediction, label in zip(forward_outputs, labels):
            batch_size += 1
            prediction = int(prediction.item() >= 0.5)
            label = int(label.item())
            if prediction == label:
                good += 1
        return good / batch_size
    
    @staticmethod
    def top_ranking(
        forward_outputs : Union[Tensor, Array],
        labels : Union[Tensor, Array],
    ) -> float:
        good, batch_size = 0, 0
        for prediction, label in zip(forward_outputs, labels):
            batch_size += 1
            prediction = prediction.argmax().item() if isinstance(prediction, Tensor) else np.argmax(prediction)
            label = label.item() if isinstance(label, Tensor) else label
            good = good + 1 if prediction == label else good
        return good / batch_size

class MAP(Metric):

    best = max
    nature = 'map'

    @classmethod
    def compute(
        cls,
        problem_type : str,
        loss : float,
        forward_outputs : Tensor,
        labels : Tensor,
    ) -> float:
        if problem_type == 'binary_classification':
            return Accuracy.binary_classification(forward_outputs, labels)
        elif problem_type == 'top_ranking':
            return cls.top_ranking(forward_outputs, labels)
        else:
            raise(ValueError(f"Unsupported problem type for MAP metric: {problem_type}"))

    @staticmethod
    def top_ranking(
        forward_outputs : Tensor,
        labels : Tensor,
    ) -> float:
        acc_map, batch_size = 0, 0
        transpositions = forward_outputs.sort(descending=True)[1]
        for i, y in enumerate(labels):
            batch_size += 1
            zeros = torch.zeros(forward_outputs[i].size())
            zeros[y.item()] = 1
            rank = zeros[transpositions[i]].argmax()+1
            acc_map += 1 / rank.item()
        return acc_map / batch_size
            
class Top_k_accuracy(Metric):

    best = max

    def __init__(
        self,
        k : int,
        phase : str,
        ) -> None:
        self.k = k
        self.nature = f'top {k} accuracy'
        self.phase = phase
        self.running = False
        self.history = []
        self.acc_measures = []
        self.acc_sizes = 0

    def compute(
        self,
        problem_type : str,
        loss : float,
        forward_outputs : Tensor,
        labels : Tensor,
        ) -> float:
        if problem_type == 'binary_classification':
            return Accuracy.binary_classification(forward_outputs, labels)
        elif problem_type == 'top_ranking':
            return self.top_ranking(forward_outputs, labels)
        else:
            raise(ValueError(f"Unsupported problem type for accuracy metric: {problem_type}"))

    def top_ranking(
        self,
        forward_outputs : Tensor,
        labels : Tensor,
    ) -> float:
        good, batch_size = 0, 0
        transpositions = forward_outputs.sort(descending=True)[1]
        for i, y in enumerate(labels):
            batch_size += 1
            zeros = torch.zeros(forward_outputs[i].size())
            zeros[y.item()] = 1
            rank = zeros[transpositions[i]].argmax()+1
            good += int(rank < self.k)
        return good / batch_size
