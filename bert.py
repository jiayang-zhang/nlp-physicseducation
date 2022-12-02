import pandas as pd
import numpy as np
from tabulate import tabulate
import random

from transformers import BertTokenizer, BertForSequenceClassification # from Hugging Face
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tools import ml_tools, utils

# =======================================================================================================================================
# * set which dataset you would like to train *
dir_csv = 'outputs/labels_cleaned_y1c1c2.csv' # csv file input path
bert_path = 'outputs/bert_model/'    # output paths
# =======================================================================================================================================
# * set how you would like it to be trained *
label = 'ReasoningLevel' # 'ArgumentLevel', 'ReasoningLevel'
# =======================================================================================================================================

# -- load corpus --
df = pd.read_csv(dir_csv, encoding='utf-8')
df = df[['StudentID', 'Content', label]]

# map label(string) to number
label_dict, value = {}, 0
for key in df[label].unique():
    label_dict.update({key: value})             # create key value pair   e.g 'prediction': 4
    df[label].replace(key, value,inplace=True)  # replace labels in dataframe
    value+=1


text = df['Content'].tolist()
labels = df[label].tolist()



# -- tokenisation --
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )


def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 512, # max length 512
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

token_id, attention_masks = [], []
for sample in text:
  encoding_dict = preprocessing(sample, tokenizer)
  token_id.append(encoding_dict['input_ids'])
  attention_masks.append(encoding_dict['attention_mask'])


token_id = torch.cat(token_id, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(labels)

# -- split data --

val_ratio = 0.2
batch_size = 16 # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf

# Indices of the train and validation splits stratified by labels
train_idx, val_idx = train_test_split(
    np.arange(len(labels)),
    test_size = val_ratio,
    shuffle = True,
    stratify = labels)

# Train and validation sets
train_set = TensorDataset(token_id[train_idx],
                          attention_masks[train_idx],
                          labels[train_idx])

val_set = TensorDataset(token_id[val_idx],
                        attention_masks[val_idx],
                        labels[val_idx])

# -- prepare dataloader --
train_dataloader = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_set,
            sampler = SequentialSampler(val_set),
            batch_size = batch_size
        )

print(val_set)


'''
reference: https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894
'''
