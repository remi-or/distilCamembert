# distilCamemBERT

This code aims to distill CamemBERT, a french model based on RoBERTa, to have a smaller and faster model with _hopefully_ equivalent performances.

## Configuration

Below is a table with the changing parameters in RoBERTa, distilRoBERTa and CamenBERT configurations, and our proposed configuration for distilCamemBERT.

| Model | architecture | bos_token_id | eos_token_id | output_past | num_hidden_layers | model_type | vocab_size |
| :---- | :----------: | :----------: | :----------: | :---------: | ----------------: |  --------: |  --------: |
| RoBERTa | RobertaForMaskedLM | 0 | 2 | Absent | 12 | "roberta" | 50265 |
| distilRoBERTa | RobertaForMaskedLM | 0 | 2 | Absent | 6 | "roberta" | 50265 |
| CamemBERT | CamembertForMaskedLM | 5 | 6 | true | 12 | "camembert" | 32005 |
| distilCamemBERT | CamembertForMaskedLM | 5 | 6 | true | 6 | "camembert" | 32005 |

As the distilRoBERTa architecture is a copy of RoBERTa's with only the number of hidden layer halved, our distilCamemBERT architecture does the same.

## Initialization

As in [DistilBERT's paper](https://arxiv.org/abs/1910.01108), we initialize our model by copying one out of two attention layer of the given CamemBERT model.
