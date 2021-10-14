# Imports
from typing import Tuple
from transformers import CamembertModel, CamembertConfig
from torch.nn import Module


# Global scope variable
distilCamemBERT_config = CamembertConfig.from_json_file('distilCamemBERT_config.json')


# Main function
def distill(
    CamemBERT : CamembertModel,
) -> CamembertModel:
    """
    Distills a given (CamemBERT) model to a smaller distilCamemBERT model as was done with BERT and DistilBERT.
    """
    distilCamemBERT = CamembertModel(distilCamemBERT_config)
    for i, (new_part, old_part) in enumerate(zip(distilCamemBERT.children(), CamemBERT.children())):
        # Embeddings or Pooler
        if i == 0 or i == 2:
            new_part.load_state_dict(old_part.state_dict())
        # Encoder
        elif i == 1:
            new_layers = [layer for layer in next(new_part.children())]
            old_layers = [layer for layer in next(old_part.children())]
            for j in range(6):
                new_layers[j].load_state_dict(old_layers[2 * j].state_dict())
        # Fail case
        else:
            raise(ValueError(f"Found more than three parts in a CamemBERT model, which isn't supposed to happen. i = {i}"))
    return distilCamemBERT


# Misc. function
def number_of_parameters(
    model : Module,
    trainable : bool = False,
) -> Tuple[int, int]:
    """
    Given a torch (model), returns the number of parameters it has.
    If the (trainable) flag is set to True, only count the trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable))