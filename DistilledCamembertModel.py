# region Imports
from typing import Union, Any, Tuple, List, Dict

import torch

from transformers.models.camembert.modeling_camembert import CamembertForSequenceClassification, CamembertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaModel

from utils.module import Module
# endregion

# region Types
SupportedCamembert = Union[
    CamembertForSequenceClassification,
    ]
Outputs = Union[
    SequenceClassifierOutput,
]
Tensor = torch.Tensor
# endregion

# region DistilledCamembert class
class DistilledCamembertModel(Module):

    """
    A class to distill a CamemBERT in this way DistilBERT did it.
    """

    def __init__(
        self,
        teacher : SupportedCamembert,
        verbose : bool = False,
        temperature : float = 1,
        ) -> None:
        super(DistilledCamembertModel, self).__init__()
        self.teacher = teacher
        self.student = self.distill(teacher, verbose)
        self.teacher.eval()
        self.temperature = temperature

    # region Temperature
    @property
    def temperature(
        self,
    ) -> float:
        return self._temperature if self.training else 1

    @temperature.setter
    def temperature(
        self,
        value : float,
    ) -> None:
        if value < 1:
            raise(ValueError(f"Temperature must be above 1, it cannot be {value}"))
        else:
            self._temperature = value
    # endregion

    def create_student(
        self,
        teacher : SupportedCamembert,
        config : CamembertConfig,
        ) -> SupportedCamembert:
        """
        Creates a student model with the same type as (teacher) with the given (config).
        """
        if isinstance(teacher, CamembertForSequenceClassification):
            student = CamembertForSequenceClassification(config)
        else:
            raise(ValueError(f"Unsupported CamemBERT type: {type(teacher)}"))
        return student

    def teach_student(
        self,
        teacher : Any,
        student : Any,
        verbose : bool = False,
        ) -> None:
        """
        Transfers half the (teacher) encoder layers to the (student), along with all the other weights.
        If the (verbose) flag is passed, the function displays the types of the teacher and the student.
        """
        if verbose:
            print(f"The teach_student method got called on these types:")
            print(f"teacher - {type(teacher)}")
            print(f"student - {type(student)}")
            print()
        # If the part is a supported CamemBERT or an entire RoBERTa model, unpack and iterate
        if isinstance(teacher, RobertaModel) or isinstance(teacher, SupportedCamembert):
            for old_part, new_part in zip(teacher.children(), student.children()):
                self.teach_student(old_part, new_part, verbose)
        # Else if the part is an encoder
        elif isinstance(teacher, RobertaEncoder):
                teacher_encoding_layers = [layer for layer in next(teacher.children())]
                student_encoding_layers = [layer for layer in next(student.children())]
                for i in range(len(student_encoding_layers)):
                    student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
        # Else the part is a regular part
        else:
            student.load_state_dict(teacher.state_dict())

    def distill(
        self,
        CamemBERT : SupportedCamembert,
        verbose : bool = False,
        ) -> SupportedCamembert:
        """
        Distills a given (CamemBERT) model to a smaller distilCamemBERT model as was done with BERT and DistilBERT.
        This means the two model share the same configuration except the number of layers.
        If the (verbose) flag is passed, one can see the details of the copying process
        """
        # Create student configuration
        distilled_config = CamemBERT.config.to_dict()
        distilled_config['num_hidden_layers'] //= 2
        distilled_config = CamembertConfig.from_dict(distilled_config)
        # Create uninitialized student model
        distilCamemBERT = self.create_student(teacher=CamemBERT, config=distilled_config)
        self.teach_student(CamemBERT, distilCamemBERT, verbose)
        return distilCamemBERT

    def ce_loss(
        self,
        teacher_pooler_outputs : Tensor,
        student_pooler_outputs: Tensor,
        ) -> Tensor:
        teacher_pooler_outputs = torch.softmax(teacher_pooler_outputs / self.temperature, dim=1)
        teacher_pooler_outputs = torch.log(teacher_pooler_outputs)
        student_pooler_outputs = torch.softmax(student_pooler_outputs / self.temperature, dim=1)
        tensor_to_minimize = torch.sum(teacher_pooler_outputs * student_pooler_outputs, dim=1)
        return torch.nn.L1Loss()(
            tensor_to_minimize,
            torch.zeros(tensor_to_minimize.size()[0], device=self.device),
        )

    def batch(
        self,
        dataset_batch : Dict[str, Union[str, Tensor]],
        ) -> Tuple[str, Tensor, Tensor]:
        problem_type, (teacher_outputs, student_outputs) = self.forward(dataset_batch)
        cosine_loss = torch.nn.CosineEmbeddingLoss()(
            teacher_outputs[0],
            student_outputs[0],
            torch.ones(teacher_outputs[0].size()[0], device=self.device),
        )
        cross_entropy_loss = self.ce_loss(
            teacher_outputs[0],
            student_outputs[0],
        )
        if problem_type == 'classification':
            specific_loss = torch.nn.CrossEntropyLoss()(
                student_outputs[1],
                dataset_batch['Y'].to(self.device),
            )
        loss = (cosine_loss + cross_entropy_loss + specific_loss) / 3
        return problem_type, student_outputs[1], loss

    def forward(
        self,
        dataset_batch : Dict[str, Union[str, Tensor]],
        ) -> Tuple[str, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        if dataset_batch['forward_type'] == 'tanda':
            teacher_outputs, student_outputs = self.forward_sequence_classification(*dataset_batch['couples'])
            problem_type = 'classification'
        else:
            raise(ValueError(f"Can't perform forward pass with teacher of type {type(self.teacher)}."))
        return problem_type, (teacher_outputs, student_outputs)

    def get_pooler_outputs(
        self, 
        input_ids : Tensor,
        attention_mask : Tensor,
        teacher : bool,
        ) -> Tuple[Tensor]:
        if teacher:
            pooler_outputs = self.teacher.roberta(input_ids, attention_mask)[0]
        else:
            pooler_outputs = self.student.roberta(input_ids, attention_mask)[0]
        return pooler_outputs

    def classfiy(
        self,
        pooler_outputs : Tensor,
        teacher : bool,
        ) -> Tensor:
        if teacher:
            return self.teacher.classifier(pooler_outputs)
        else:
            return self.student.classifier(pooler_outputs)

    def forward_sequence_classification(
        self,
        input_ids : Tensor,
        attention_mask : Tensor,
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        # Teacher part, no gradient 
        with torch.no_grad():
            teacher_pooler_outputs = self.get_pooler_outputs(input_ids.to(self.device), attention_mask.to(self.device), teacher=True)
            teacher_classifier_outputs = self.classfiy(teacher_pooler_outputs, teacher=True)
            teacher_outputs = (teacher_pooler_outputs[:, 0, :], teacher_classifier_outputs)
        # Student part, with gradient (except if already in a no gradient context)
        student_pooler_outputs = self.get_pooler_outputs(input_ids.to(self.device), attention_mask.to(self.device), teacher=False)
        student_classifier_outputs = self.classfiy(student_pooler_outputs, teacher=False)
        student_outputs = (student_pooler_outputs[:, 0, :], student_classifier_outputs)
        return teacher_outputs, student_outputs