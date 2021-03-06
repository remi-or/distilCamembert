# region Imports
from __future__ import annotations

from typing import Optional, Dict, Union, List, Tuple

import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

import re
import os
import json
import torch

from transformers import CamembertTokenizer
# endregion

# region Custom types
Array = np.ndarray
Dataframe = pd.DataFrame
Tensor = torch.Tensor

Path = str
Tokens = List[int]
Tokenizer = Union[CamembertTokenizer]
Batch = Dict[str, Union[str, Tensor]]
# endregion

# region Misc. function
def pad_and_tensorize(
    tensor_as_list : List[Tokens],
    return_attention_mask : bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Pads and tensorizes a list of Tokens, where List[Tokens] = List[List[int]].

    Inputs:
        - (tensors_as_list), a list of list of int
        - (return_attention_mask), a flag to return the attention masks associated with the tensors

    Outputs:
        - a Tensor: the resulting tensor
        - a Tensor: its attention mask, if asked for
    """
    max_length = max(len(list) for list in tensor_as_list)
    tensor, attention_masks = [], []
    for list in tensor_as_list:
        padding = max_length - len(list)
        tensor.append(list + [0] * padding)
        if return_attention_mask:
            attention_masks.append([1] * len(list) + [0] * padding)
    if return_attention_mask:
        return torch.tensor(tensor), torch.tensor(attention_masks)
    else:
        return torch.tensor(tensor)

def count(
    list_of_elements : List[int],
    ) -> Dict[int, int]:
    """
    Counts the number of occurences of each elements in a (list_of_elements).
    Returns it as a dictionary with the element as a key and the number of occurence as a value.
    """
    acc = {element : 0 for element in list_of_elements}
    for element in list_of_elements:
        acc[element] += 1
    return acc
# endregion

# region from_squad_like related functions
def remove_abreviations(
    text : str,
) -> str:
    """
    Removes abbreviations from a (text).
    """
    text = re.sub('[A-Z]\.', lambda m: m.group(0)[0].lower(), text)
    text = text.replace('art.', 'article')
    text = text.replace('r??f.', 'r??f??rence')
    text = text.replace('etc.', 'etc')
    text = text.replace('av.', 'avant')
    text = text.replace('av.', 'avant')
    text = text.replace('hab.', 'habitants')
    text = text.replace('Sr.', 'Senior')
    text = text.replace('??c', 'degr??s Celsius')
    text = text.replace('??C', 'degr??s Celsius')
    return text

def extract_sentences(
    paragraph : dict,
    min_length : int = 3,
) -> List[str]:
    """
    Extracts sentences from a (squad_like_datum) which length is superior to (min_length).
    """
    text_acc, already_added = '', set()
    text = remove_abreviations(paragraph['context'])
    if text not in already_added:
        text_acc = text_acc + ' ' + text
        already_added.add(text)
    sentences = []
    for sentence in text.split('.'):
        sentence = sentence.strip()
        if len(sentence) > min_length:
            sentences.append(sentence + '.')
    return sentences

def extract_qas(
    paragraph : dict,
    sentences : List[str],
    not_found : int,
) -> Tuple[List[str], List[int]]:
    questions, ids, not_found = [], [], 0
    for qas in paragraph['qas']:
        questions.append(qas['question'])
        answer = qas['answers'][0]['text']
        answer = remove_abreviations(answer).strip()
        for i, sentence in enumerate(sentences):
            if answer in sentence:
                ids.append(i)
                break
        if len(questions) != len(ids):
            not_found += 1
            questions.pop()
    return questions, ids, not_found
# endregion

class QaDataset:

    """
    A class to implement QaDatasets, which are composed of questions and passages.
    For now, each question can only have one answer.
    """

    # region Magic methods
    def __init__(
        self,
        ) -> None:
        """
        This is a dummy function, QaDatasets are always supposed to be initialized from a staticmethod.
        """
        self.questions = pd.DataFrame({'text' : [], 'answer_id' : [], 'theme' : []})
        self.passages = pd.DataFrame({'text' : [], 'id' : [], 'theme' : []})
        self._batch_type = None
        self.tokenized = False

    def __len__(
        self,
        ) -> int:
        """
        The length of the dataset is the number of questions it holds.
        """
        return self.questions.shape[0]

    def __add__(
        self,
        other : QaDataset,
        ) -> QaDataset:
        """
        Adds two QaDataset and returns the result, without modifying any of the two input.
        """
        # Copy to avoid changing the inputs
        self = self.copy()
        other = other.copy()
        # Shift the passages' indexes to avoid indexing conflicts
        n = max(self.passages.loc[:, 'id']) + 1
        for i, row in other.questions.iterrows():
            other.questions.loc[i, 'answer_id'] = row['answer_id'] + n
        for i, row in other.passages.iterrows():
            other.passages.loc[i, 'id'] = row['id'] + n
        # Add the questions and passages
        self.questions = self.questions.append(other.questions).reset_index(drop=True)
        self.passages = self.passages.append(other.passages).set_index('id', drop=False)
        # Update the tokenized flag
        self.tokenized = self.tokenized and other.tokenized
        # Update the batch type
        self._batch_type = None
        # Compact the result
        self.compact()
        return self

    def __getitem__(
        self,
        i : int,
        ) -> Tuple[str, str]:
        """
        Retrieve the question with id (i) and its associated passage.
        """
        question = self.questions.loc[i, 'text']
        answer_id = self.questions.loc[i, 'answer_id']
        return question, self.passages.loc[answer_id, 'text']
    # endregion

    # region Internal storage methods
    def compact(
        self,
        ) -> None:
        """
        Compacts the QaDataset by updating the passages' ids so it doesn't have holes.
        """
        conversion = {}
        for i, id in enumerate(self.passages.loc[:, 'id']):
            conversion[id] = i
        self.passages.loc[:, 'id'] = [i for i in range(self.passages.shape[0])]
        self.passages.set_index('id', drop=False)
        for i, row in self.questions.iterrows():
            self.questions.loc[i, 'answer_id'] = conversion[row['answer_id']]

    def copy(
        self,
        deep : bool = True,
        ) -> QaDataset:
        """
        Returns a copy, or a deepcopy, of the QaDataset if the (deep) flag is passed.
        """
        return QaDataset.from_existing(self.questions.copy(deep), self.passages.copy(deep), self._batch_type, self.tokenized)

    def split(
        self,
        p : float,
        seed : Optional[int] = None,
        ) -> Tuple[QaDataset, QaDataset]:
        """
        Splits the dataset's questions into two parts, the first with the size of (p)*the dataset size.
        Repeatability can be ensured with (seed).
        """
        cutoff = int(p * self.questions.shape[0])
        self.questions = self.questions.sample(frac=1, random_state=seed)
        questions_df1, questions_df2 = self.questions.loc[:cutoff, :], self.questions.loc[cutoff:, :]
        return (
            QaDataset.from_existing(questions_df1, self.passages, self._batch_type, self.tokenized),
            QaDataset.from_existing(questions_df2, self.passages, self._batch_type, self.tokenized),
        )

    def shuffle(
        self,
        seed : Optional[int] = None,
        ) -> None:
        """
        Shuffle the questions. Repeatability can be ensured with (seed).
        """
        rng = np.random.default_rng(seed)
        index = rng.permutation(len(self.X))
        self.X = self.X[index]
        self.Y = self.Y[index]

    def as_test(
        self,
        ) -> Tuple[List[str], List[str], Tensor]:
        """
        Returns the list of questions, passages and the correct score for a top_ranking model.
        """
        questions = list(self.questions.loc[:, 'text'])
        passages = list(self.passages.loc[:, 'text'])
        target = np.zeros((len(questions), len(passages)))
        for i, row in self.questions.iterrows():
            target[i, row['answer_id']] = 1
        return questions, passages, torch.from_numpy(target)

    def save(
        self,
        path : str,
        overwrite : bool = False,
        ) -> None:
        """
        Saves the QaDataset object to a given (path). If something already exists there, (overwrite) must be turned on.
        """
        already_exists = os.path.exists(path + '.json')
        if not already_exists or overwrite:
            state_dict = {}
            state_dict['questions'] = self.questions.to_json()
            state_dict['passages'] = self.passages.to_json()
            state_dict['_batch_type'] = self._batch_type
            state_dict['tokenized'] = self.tokenized
            with open(path, mode='w', encoding='utf-8') as file:
                json.dump(state_dict, file)
    # endregion

    # region @staticmethods to create a QaDataset
    @staticmethod
    def from_saved(
        path : str,
        ) -> QaDataset:
        """
        Loads a QaDataset from a previous save.
        """
        instance = QaDataset()
        with open(path, mode='r', encoding='utf-8') as file:
            state_dict = json.load(file)
        instance.questions = pd.read_json(state_dict['questions'])
        instance.passages = pd.read_json(state_dict['passages'])
        instance._batch_type = state_dict['_batch_type']
        instance.tokenized = state_dict['tokenized']
        return instance

    @staticmethod
    def from_existing(
        questions : Dataframe,
        passages : Dataframe,
        _batch_type : Optional[str] = None,
        tokenized : bool = False,
        ) -> QaDataset:
        """
        Create a QaDataset from dataframes of (questions) and (passages), and eventualy a (_batch_type) and (tokenized) argument.
        """
        new_instance = QaDataset()
        new_instance.questions = questions.reset_index(drop=True)
        new_instance.passages = passages.set_index('id', drop=False)
        new_instance._batch_type = _batch_type
        new_instance.tokenized = tokenized
        new_instance.compact()
        return new_instance

    @staticmethod
    def from_squad_like(
        squad_like : Union[Path, dict],
        ) -> QaDataset:
        if isinstance(squad_like, Path):
            with open(squad_like, encoding='utf-8') as file:
                squad_like = json.load(file)['data']
        questions, answer_ids, sentences, question_themes, sentence_themes, not_found = [], [], [], [], [], 0
        for datum in squad_like:
            theme = datum['title']
            for paragraph in datum['paragraphs']:
                paragraph_sentences = extract_sentences(paragraph)
                paragraph_questions, paragraph_answer_ids, paragraph_not_found = extract_qas(paragraph, paragraph_sentences, not_found)
                questions += paragraph_questions
                question_themes += [theme] * len(paragraph_questions)
                answer_ids += [len(sentences) + answer_id for answer_id in paragraph_answer_ids]
                sentences += paragraph_sentences
                sentence_themes += [theme] * len(paragraph_sentences)
                not_found += paragraph_not_found
        print(f"During data loading, {not_found} questions were not found out of {len(questions)}, which is {not_found / len(questions)} of loss.")
        return QaDataset.from_existing(
            pd.DataFrame({
                'text' : questions,
                'answer_id' : answer_ids,
                'theme' : question_themes,
            }),
            pd.DataFrame({
                'text' : sentences,
                'id' : list(range(len(sentences))),
                'theme' : sentence_themes,
            })
        )
    # endregion

    def tokenize(
        self,
        tokenizer : Tokenizer,
        max_length : int = 512,
        ) -> None:
        """
        Adds a tokenized representation for each question and passage.
        """
        if self.tokenized:
            print('This dataset was already tokenized!')
            return None
        self.tokenized = True
        tokenizer_kwargs = {
            'max_length' : max_length,
            'padding' : False,
            'truncation' : True,
        }
        # Converts X data from strings to token_lists
        self.questions.loc[:, 'tokenized'] = tokenizer(list(self.questions.loc[:, 'text']), **tokenizer_kwargs)['input_ids']
        self.passages.loc[:, 'tokenized'] = tokenizer(list(self.passages.loc[:, 'text']), **tokenizer_kwargs)['input_ids']

    batch_types = {
        0 : 'group_by_theme',
        1 : 'tanda',
    }

    def set_batch_type(
        self,
        id : int,
        ) -> None:
        """
        Sets whitch kind of batch the dataset delivers.
        """
        if not self.tokenized:
            raise(ValueError("Please tokenize the dataset first."))
        if id == 0:
            self._batch_type = self.batch_types[0]
            self.build_group_by_theme()
        elif id == 1:
            self.build_tanda()
            self._batch_type = self.batch_types[1]
        else:
            raise(ValueError(f"Unknown forward type."))

    def batches(
        self,
        # General arguments
        shuffle : bool = True, 
        batch_size : int = 1,
        # group_by_theme arguments
        fill : bool = False,
        gold_percentage : float = 0.0,
        # tanda arguments
        sep_token : int = 6,
        force_equity : bool = True,
        ) -> Batch:
        """
        Delivers batches according to the QaDataset configuration.
        """
        if shuffle:
            self.shuffle()
        if not self.tokenized:
            raise(ValueError("Can't yield batches from an untokenized dataset!"))
        if self._batch_type == 'group_by_theme':
            return self.batches_group_by_theme(
                gold_percentage=gold_percentage,
                trim=batch_size,
                fill=fill,
                )
        elif self._batch_type == 'tanda':
            return self.batches_tanda(
                batch_size=batch_size,
                sep_token=sep_token,
                force_equity=force_equity,
            )
        else:
            raise(ValueError("Can't yield batches from a dataset with no forward type!"))

    def diagnose(
        self,
        ) -> None:
        """
        Plots various information about the QaDataset. Must be already tokenized.
        """
        if not self.tokenized:
            raise(ValueError("Please tokenize this dataset first."))
        for _i, df in enumerate([self.questions, self.passages]):
            lengths = count([len(x) for x in df.loc[:, 'tokenized']])
            xs, ys = [length for length in lengths], [lengths[length] for length in lengths]
            plt.plot(xs, ys, 'bo')
            plt.title(f"Size of tokenized {['questions', 'passages'][_i]} in the dataset")
            plt.xlabel('Size')
            plt.ylabel(f"Nb. of {['questions', 'passages'][_i]}")
            plt.show()
            ypercent = np.zeros((len(ys), ))
            for i in range(ypercent.shape[0]):
                ypercent[i] = sum(ys[:i+1])
            ypercent /= sum(ys)
            ypercent = 1 - ypercent
            ypercent *= 100
            plt.plot(xs, ypercent)
            plt.title(f"Percent of tokenized {['questions', 'passages'][_i]} in the dataset length inferior to size")
            plt.xlabel('Size')
            plt.ylabel(f"% of {['questions', 'passages'][_i]}")
            plt.show()

    # region group_by_theme
    def build_group_by_theme(
        self,
        ) -> None:
        self.X, self.Y = [], []
        themes = list(set(self.questions.loc[:, 'theme']))
        for theme in themes:
            acc_questions, acc_passages, acc_y = [], [], []
            for _, row in self.questions.loc[self.questions.loc[:, 'theme'] == theme, :].iterrows():
                acc_questions.append(row['tokenized'])
                good_passage = self.passages.loc[row['answer_id'], 'tokenized']
                if good_passage not in acc_passages:
                    acc_passages.append(good_passage)
                    acc_y.append(len(acc_passages)-1)
                else:
                    for i, passage in enumerate(acc_passages):
                        if passage == good_passage:
                            acc_y.append(i)
                            break
                nb_good_passages = len(acc_passages)
            for passage in self.passages.loc[self.passages.loc[:, 'theme'] == theme, 'tokenized']:
                if passage not in acc_passages:
                    acc_passages.append(passage)
            self.X.append([acc_questions, acc_passages, nb_good_passages, theme])
            self.Y.append(acc_y)

    def batches_group_by_theme(
        self,
        gold_percentage : float = 0.0,
        trim : Optional[int] = None,
        fill : bool = False,
        ) -> Batch:
        for X, Y in zip(self.X, self.Y):
            trim = len(X[1]) if trim is None else trim
            questions, passages, nb_good_passages, theme = X
            batch = {'forward_type' : 'gold'}
            batch['questions'] = pad_and_tensorize(questions[:trim])
            passages, misleading_passages = passages[:nb_good_passages], passages[nb_good_passages:]
            if nb_good_passages > trim:
                passages = passages[:trim]
            else:
                gold_passages_to_add = int(gold_percentage * len(misleading_passages))
                while gold_passages_to_add > 0 and len(passages) < trim:
                    passages.append(self.choose_random_passage(theme))
                    gold_passages_to_add -= 1
                if len(passages) < trim:
                    rd.shuffle(misleading_passages)
                    while len(passages) < trim and misleading_passages:
                        passages.append(misleading_passages.pop())
                    while len(passages) < trim and fill:
                        passages.append(self.choose_random_passage(theme))
            batch['passages'] = pad_and_tensorize(passages)
            batch['Y'] = torch.tensor(Y[:trim])
            yield batch

    def choose_random_passage(
        self,
        theme_to_exclude : Optional[str] = None,
        ) -> Tokens:
        if theme_to_exclude is None:
            chosen_id = rd.randint(0, self.passages.shape[0]-1)
        else:
            chosen_id = -1
            while chosen_id == -1:
                chosen_id = rd.randint(0, self.passages.shape[0]-1)
                if self.passages.loc[chosen_id, 'theme'] == theme_to_exclude:
                    chosen_id = -1
        return self.passages.loc[chosen_id, 'tokenized']

    # endregion

    # region tanda
    def build_tanda(
        self,
        ) -> None:
        self.X = [i for i in range(len(self))]
        self.Y = []
        self.masks = {theme : (self.passages.loc[:, 'theme'] == theme) for theme in set(self.passages.loc[:, 'theme'])}
        self.masks_sum = {theme : mask.sum() for theme, mask in self.masks.items()}

    def batches_tanda(
        self,
        batch_size : int = 1,
        sep_token : int = 6,
        force_equity : bool = True,
        ) -> Batch:
        for i_batch in range(0, len(self.X), batch_size):
            batch = {'forward_type' : 'tanda'}
            couples, labels, indexes = [], [], self.X[i_batch : i_batch + batch_size]
            for i in indexes:
                theme = self.questions.loc[i, 'theme']
                mask = self.masks[theme]
                nb_possible_passages = self.masks_sum[theme]
                chosen = rd.randint(1, 2 if force_equity else nb_possible_passages)
                # Right passage (meaning real answer) case
                if chosen == 1 or nb_possible_passages == 1:
                    passage = self.passages.loc[self.questions.loc[i, 'answer_id'], 'tokenized'] 
                # Wrong passage (meaning not the answer) case
                else:
                    mask[self.questions.loc[i, 'answer_id']] = False
                    passage = self.passages.loc[mask, 'tokenized'].sample(n=1).iloc[0]
                    mask[self.questions.loc[i, 'answer_id']] = True
                labels.append(int(chosen == 1))
                passage[0] = sep_token
                couples.append(self.questions.loc[i, 'tokenized'] + passage)
            batch['couples'] = pad_and_tensorize(couples, return_attention_mask=True)
            batch['Y'] = torch.tensor(labels)
            yield batch               
    # endregion

