import pandas as pd
from transformers import BertTokenizer
# ============================================================
dir_csv = 'outputs/sections/labels_cleaned_y1c1c2.csv'
# ============================================================

# load the dataset into a pandas dataframe
df = pd.read_csv(
        dir_csv,
        encoding='utf-8',
        skiprows = 1,
        names=['StudentID', 'Content', 'ArgumentLevel', 'ReasoningLevel']
)





# define dict to code labels to numbers
ReasoningLevel_dict = {'bal': 0, 'the': 1, 'exp': 2, 'none': 3}
ArgumentLevel_dict = {'extended': 0, 'deep': 1, 'expert': 2, 'superficial': 3, 'prediction': 4}

# replace to number labels
df['ReasoningLevel'].replace(list(ReasoningLevel_dict.keys()), list(ReasoningLevel_dict.values()),inplace=True)
df['ArgumentLevel'].replace(list(ArgumentLevel_dict.keys()), list(ArgumentLevel_dict.values()),inplace=True)

# to numpy ndarrys
corpus = df.Content.values
ArgumentLevels = df.ArgumentLevel.values
ReasoningLevels = df.ReasoningLevel.values
print('Number of reports: {:,}\n'.format(len(corpus)))
print('First report:    ',corpus[0])
print('Second report:   ',corpus[1])





# Load the BERT tokenizer
print('loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
