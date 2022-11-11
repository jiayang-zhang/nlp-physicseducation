'''
performs:
tf-ifd
naive_bayes
'''

from tools.xlsxer import *
from tools.xmler import *
import os
import pandas as pd
pd.set_option('max_colu', 10)
from sklearn.feature_extraction.text import CountVectorizer


# =======================================================================================================================================
dir_txtfldr = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
# =======================================================================================================================================


# -- Get files ---
df_files = build_files_dataframe(dir_txtfldr, 'GS_', '.txt')

# -- Get labels ---
df_labels = build_labels_dataframe('data/labels.xlsx')

# -- Merge dataframes --
df = pd.merge(df_files, df_labels, left_on='StudentID', right_on='StudentID')      # merged dataframe: StudentID, Content, ArgumentLevel, ReasoningLevel


"""
Examples
"""
# To call 'ArgumentLevel' labels
ArgumentLevel_list = print(df['ArgumentLevel'].tolist())
# To see the corresponding document names
print(df['StudentID'].tolist())
# To call of string lists of reports for feature extraction
print(df['Content'].tolist())
# To see everything
print(df)



# -- Feature extraction: TF-IDF ---
# word vectors
corpus_wordvec_names = None                    # word vector name list
corpus_wordvec_counts = None                   # word vector frequncy list    # len(corpus_wordvec_counts) = index(dataframe)
