from tools import ml_tools, utils
import os
import pandas as pd

# who developed transformers packages?
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup



# =======================================================================================================================================
# input paths
dir_csv = 'outputs/labels_cleaned.csv'

# output paths
bert_path = 'outputs/bert_model/'    # TODO: 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
# =======================================================================================================================================

df = pd.read_csv(dir_csv, encoding='utf-8')

# find max length of text from one documents
'''
maxlen = 0
for list in df['Content'].tolist():
    if len(list) > maxlen:
        maxlen = len(list)
print(maxlen)
'''

# -- pre-process data --
tokenizer = BertTokenizer.from_pretrained(bert_path)   # initialise tokenizer
input_ids, input_masks, input_types,  = [], [], []  # TODO: input char ids,  segment type ids,  attention mask
labels = []
maxlen = 22552     # max lengths of text from one documents
