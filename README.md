[DistilBERT]: https://arxiv.org/abs/1910.01108
[distilRoBERTa]: https://huggingface.co/distilroberta-base

# distilCamemBERT

This code aims to distill CamemBERT, a french model based on RoBERTa, to have a smaller and faster model with _hopefully_ equivalent performances.

## Motivation

Although CamemBERT is a solid model when it comes to french NLP tasks, it is quite a large model, with 110M parameters. This is a little short of RoBERTa's number of parameters, which is 125M. This difference is due to the variation in RoBERTa vocab_size of 50625 and CamemBERT's, which is 32005.  
Since CamemBERT is based on RoBERTa, we looked for existing distilation of this model. We found [distilRoBERTa][distilRoBERTa], which has 88M parameters and claims to be roughly twice as fast as RoBERTa. It has less parameters than the smallest BERT that's cased (as RoBERTa is cased) BERT, BERT-base-cased which has 109M parameters. It also largely outperforms BERT on the GLUE test.  
Thus, the most promising route to having a smaller and faster french model with competing performances seems to be distilRoBERTa's approach.

## Configuration

Below is a table with the changing parameters in RoBERTa, distilRoBERTa and CamenBERT configurations, and our proposed configuration for distilCamemBERT.

| Model | architecture | bos_token_id | eos_token_id | output_past | num_hidden_layers | model_type | vocab_size |
| :---- | :----------: | :----------: | :----------: | :---------: | ----------------: |  --------: |  --------: |
| RoBERTa | RobertaForMaskedLM | 0 | 2 | Absent | 12 | "roberta" | 50265 |
| distilRoBERTa | RobertaForMaskedLM | 0 | 2 | Absent | 6 | "roberta" | 50265 |
| CamemBERT | CamembertForMaskedLM | 5 | 6 | true | 12 | "camembert" | 32005 |
| distilCamemBERT | CamembertForMaskedLM | 5 | 6 | true | 6 | "camembert" | 32005 |

As the distilRoBERTa architecture is a copy of RoBERTa's with only the number of hidden layer halved, our distilCamemBERT architecture does the same. The resulting model size is **68M parameters**.

## Initialization

As in [DistilBERT's paper][DistilBERT], we initialize our model by copying one out of two attention layer of the given CamemBERT model.
