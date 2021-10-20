# Imports
from typing import Optional, Tuple, Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from utils.logger import Logger
from utils.caller import Caller

from tqdm.auto import tqdm
from time import perf_counter
import numpy as np
import random as rd
from subprocess import check_output


# Custom types
Array = np.ndarray
Dataset = Any


# Module class
class Module(torch.nn.Module):

    """
    The regular torch.nn.Module fitted with extra methods so that it works well with the rest of the implementation.
    """

    device = 'cpu'
    _safe = False
    _monitor = False

    def flush_gpu(
        self,
        text : str = '',
        ) -> None:
        """
        Frees memory on the GPU by deleting some variables and freeing the cache.
        """       
        torch.cuda.empty_cache() 
        torch.cuda.ipc_collect()
        if self._monitor:
            result = check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'], encoding='utf-8')
            memory_footprint = [int(x) for x in result.strip().split('\n')][0]
            print(text, memory_footprint)

    def cuda(
        self,
        ) -> None:
        """
        Classic cuda method to switch the model to GPU.
        Also changes the module's device.
        """
        self.device = 'cuda'
        super().cuda()

    def safe(
        self,
        state : bool = True,
        monitor : bool = False,
        ) -> None:
        """
        Turns on or off the safe mode depending on the (state) flag. 
        Also monitors the memory usage and other factors if (monitor) is set to True.
        Safe mode tries to minimze the GPU memory used during training to avoid a CUDA out of memory error.
        However, it is slower.
        """
        self._safe = state
        self._monitor = monitor

    def fit(
        self,
        optimizer : Optimizer,
        logger : Logger,
        epochs : int,
        train_ds : Dataset,
        val_ds : Optional[Dataset] = None,
        dataset_kwargs : dict = {},
        callbacks : Optional[Caller] = None,
        verbose : bool = False,
        ) -> None:
        """
        Fits the model on the given data.
        Inputs:
            - optimizer, the optimizer that will minimize the loss function,
            - logger, a Logger object to log in the stats,
            - epochs, the number of epochs of the training,
            - train_ds, the Dataset object containing the training data
            _ val_ds, the Dataset object containing the validation data
            _ dataset_kwargs, a dictionnary containing keywords args for the Datsets objects, such as batch_size
            _ callbacks, a Callbacks object with callbacks such as checkpoints or early stopping
            _ verbose, a boolean that determines wether or not stats for each epoch are printed along the way
        Outputs: None
        """
        t0 = perf_counter()
        nb_train_batches, nb_val_batches = self.get_length(train_ds, dataset_kwargs), self.get_length(val_ds, dataset_kwargs)
        if verbose:
            print(f"Iterating through both datasets once takes {perf_counter() - t0}")
        # Epoch loop
        for i_epoch in range(epochs):
            eta = np.mean(logger.times) * (epochs - i_epoch) if logger.times else '?'
            eta = "%.2f" % eta if isinstance(eta, float) else eta
            prefix = f"Epoch {i_epoch+1}/{epochs} - ETA: {eta} - "
            # Training
            self._train(optimizer, logger, train_ds, dataset_kwargs, nb_train_batches, prefix)
            # Validation
            if val_ds is not None:
                self._val(logger, val_ds, dataset_kwargs, nb_val_batches, prefix)
            # Progress display
            if verbose:
                print(f"Epoch {i_epoch+1}/{epochs}:")
                for name, m in logger.metrics.items():
                    print(f"{name}: {m[-1]}")
            # Callbacks
            if callbacks is not None:
                stop = callbacks(self, optimizer, logger, i_epoch, epochs)
                if stop:
                    break

    def get_length(
        self,
        dataset : Optional[Dataset],
        dataset_kwargs : dict,
        ) -> int:
        """
        Given a (dataset) and (dataset_kwargs), computes how much batches there are in the dataset.
        """
        length = 0
        dataset_kwargs['shuffle'] = False
        if dataset is None:
            return 0
        for _ in dataset.batches(**dataset_kwargs):
            length += 1
        return length

    def _train(
        self,
        optimizer : Optimizer,
        logger : Logger,
        train_ds : Dataset,
        dataset_kwargs : dict,
        nb_train_batches : int,
        prefix : str,
        ) -> None:
        """
        Hidden method to train the model.
        Inputs:
            - nb_train_batches, an integer representing the number of batches in train_ds for the tqdm bar,
            - prefix, a string containing the ETA for the tqdm bar,
            - for informations about other arguments, check the .fit method,
        Outputs: None.
        """
        self.train()
        logger.train()
        for ds_batch in tqdm(train_ds.batches(**dataset_kwargs), total=nb_train_batches, leave=False, desc=prefix+'Training'):
            problem_type, Z, loss = self.batch(ds_batch) 
            logger.log_batch(problem_type, float(loss), Z, ds_batch['Y'])
            if self._safe:
                del Z, ds_batch
                self.flush_gpu('Batch logging')
            optimizer.zero_grad()
            loss.backward()
            if self._safe:
                del loss
                self.flush_gpu('Backward pass')
            optimizer.step()
        logger.end()

    def _val(
        self,
        logger : Logger,
        val_ds : Dataset,
        dataset_kwargs : dict,
        nb_val_batches : int,
        prefix : str,
        ) -> None:
        """
        Hidden method to do validation on the model.
        Inputs:
            - nb_val_batches, an integer representing the number of batches in val_ds for the tqdm bar,
            - prefix, a string containing the ETA for the tqdm bar,
            - for informations about other arguments, check the .fit method,
        Outputs: None.
        """
        self.eval()
        logger.val()
        with torch.no_grad():
            for ds_batch in tqdm(val_ds.batches(**dataset_kwargs), total=nb_val_batches, leave=False, desc=prefix+'Validation'):
                problem_type, Z, loss = self.batch(ds_batch)
                logger.log_batch(problem_type, float(loss), Z, ds_batch['Y'])
        logger.end()

    def batch(
        self,
        ds_batch : dict,
        ) -> Tuple[str, Tensor, Tensor]:
        """
        Performs a forward pass on the data (ds_batch) and computes the loss.
        Outputs:
            - problem_type, a string describing the problem such as 'binary_classification', 'multiclass_classification',... , 
            - z, the tensor output of forward
            - loss, the tensor output of the loss function,
        """
        problem_type, z = self.forward(ds_batch)
        if self._safe:
            self.flush_gpu('Forward')
        loss = self.loss(z, ds_batch['Y'].to(self.device))
        if self._safe:
            self.flush_gpu('Loss computation')
        return problem_type, z, loss

    def diagnose(
        self,
        dataset : Dataset,
        dataset_kwargs : dict = {},
        verbose : bool = False,
        ) -> Tuple[dict, str, Tensor, Tensor]:
        """
        Given a (dataset) and (dataset_kwargs) for iterating over it,
        returns a random batch in that dataset, along with what a forward pass and a loss computation would return.
        Print the various outputs along the way if (verbose) is set to True.
        """
        nb_batches = self.get_length(dataset, dataset_kwargs)
        chosen_batch = rd.randint(0, nb_batches - 1)
        for i, batch in enumerate(dataset.batches(**dataset_kwargs)):
            if i == chosen_batch:
                break
        if verbose:
            print('Batch:\n', batch, '\n')
        if verbose:
            problem_type, z = self.forward(batch)
            print(problem_type, '\n', z, '\n')
        problem_type, z, loss = self.batch(batch)
        if verbose:
            print(loss)
        return batch, problem_type, z, loss