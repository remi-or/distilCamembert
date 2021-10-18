# Imports
from typing import Tuple

from torch import Module


# Function to count the number of parameters
def number_of_parameters(
    model : Module,
    trainable : bool = False,
) -> Tuple[int, int]:
    """
    Given a torch (model), returns the number of parameters it has.
    If the (trainable) flag is set to True, only count the trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable))