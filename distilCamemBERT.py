from transformers import CamembertModel, CamembertConfig

distilCamemBERT_config = CamembertConfig.from_json_file('distilCamemBERT_config.json')

def distill(
    CamemBERT : CamembertModel,
) -> CamembertModel:
    """
    Distills a given (CamemBERT) model to a smaller distilCamemBERT model as was done with BERT and DistilBERT.
    """
    distilCamemBERT = CamembertModel(distilCamemBERT_config)
    for i, (new_part, old_part) in enumerate(zip(distilCamemBERT.children(), CamemBERT.children())):
        # Embeddings
        if i == 0:
            for new_layer, old_layer in zip(new_part.children(), old_part.children()):
                new_layer.load_state_dict(old_layer.state_dict())
        # Pooler
        elif i == 3:
            for new_layer, old_layer in zip(new_part.children(), old_part.children()):
                new_layer.load_state_dict(old_layer.state_dict())
        # Encoder
        elif i == 2:
            for j, (new_layer, old_layer) in enumerate(zip(new_part.children(), old_part.children())):
                # We only copy one layer out of two
                if j % 2 == 0:
                    new_layer.load_state_dict(old_layer.state_dict())
        # Fail case
        else:
            raise(ValueError("Found more than three parts in a CamemBERT model, which isn't supposed to happen"))
    if i == 2:
        return distilCamemBERT
    else:
        raise(ValueError(f"Found only {i+1} part in the CamemBERT model, where there are supposed to be 3."))