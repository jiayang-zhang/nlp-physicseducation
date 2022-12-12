#%%
from bs4 import BeautifulSoup as bs
import os
import string
import pandas as pd
import numpy as np
pd.set_option('max_colu', 10)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import time
import pickle


#%%
'''
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
'''

# ========================================================================================
# for .txt files
# ========================================================================================

def build_files_dataframe(dir_txtfldr, str_start, str_end) -> pd.DataFrame:
    '''
    compile name and contents of txt files to a dataframe

    index       StudentID            Content
        0       GS_xxx_Redacted      xyz


    inputs --
        path:           dir to txt folder
        str_start:      shared string at the start of the filename.  e.g. 'GS_'
        str_end:        shared string at the end of the filename.  e.g. '.txt'

    returns --
        df_files:       dataframe ['StudentID', 'Content']
    '''
    df_files = pd.DataFrame(columns=['StudentID','Content'])

    for file in os.listdir(dir_txtfldr):
        if file.startswith(str_start) and file.endswith(str_end):

            # filename for dataframe
            filename = file.rsplit('.', maxsplit=2)[0]

            # file content for dataframe
            with open(os.path.join(dir_txtfldr, file), 'r') as f:
                content = f.read()
            content = preprocess(content)

            # add new row to DataFrame
            df_files.loc[len(df_files)] = [filename, content]

    return df_files


def preprocess(text):
    '''
    splits string into tokens
        (and removes punctuation, stopword tokens)

    input --
        text: a string of words

    returns --
        text: a string of words

    notes: words such as 'can't' will be destroyed into pieces (such as can and t) if you remove punctuation before tokenisation
    ref: https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    '''

    # 1. lowercase
    text = text.lower()

    # tokenise
    tokens = word_tokenize(text)
    # print(tokens)

    # 2. remove punctuation tokens
    tokens = list(filter( (lambda t: t not in string.punctuation), tokens))
    # print(tokens)

    # 3. remove stop words tokens
    # tokens = list(filter( (lambda t: t not in stopwords.words('english')), tokens ))

    # 4. lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]

    # token back to text
    text = ' '.join(str(x) for x in tokens)

    # TO DO: consider words like 'eighty-seven'
    return text  # change to text


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

#%%
# ========================================================================================
# for .xml files
# ========================================================================================


def xml_to_txt(dir_xml, dir_txt, filename, alltext = True, processtext = True):

    '''
    alltext --
        True:   get text of entire report
        False:  get paragraphs in sections in and after Results only

    preprocesstext --
        True:   preprocess() is called --> remove punctuations etc.
        False:  preprocess() is not called
    '''

    # get text output
    if alltext:
        out = getall(dir_xml, filename)
    else:
        out = getparagraphs(dir_xml, filename)


    # write output to .txt file
    if out == []: # if file is empty
        print('WARNING: '+ filename + '.xml file is empty')
        pass
    else:
        try:
            with open(os.path.join(dir_txt, filename+'.txt'), 'w') as file:
                content = ' '.join(out)
                if processtext:
                    file.write(preprocess(content))
                else:
                    file.write(content)
        except FileNotFoundError:
            print("The .txt directory does not exist")

    return


def getparagraphs(dir_xml, filename):
    '''
    Get the paragraph next to Results <head>
    '''

    # Read .xml file
    try:
        # Parse xml as txt
        with open(os.path.join(dir_xml, filename+'.tei.xml'), 'r') as file:
            content = file.read()

    except FileNotFoundError:
        print(("The .xml directory does not exist"))

    # Parse xml
    soup = bs(content, 'xml')


    # extract <head> and <p>
    out = []
    for head in soup.select('head'):

        # check lowercase <head>, works with "RESULTS AND"
        if 'result' in head.get_text().lower():

            # Get the paragraph next to <head>
            s = head.next_sibling
            if s == None:
                print('No text found...')
                print('error filename:',filename)
            else:
                out.append(s.get_text(' ', strip=True)) #string

            # TO DO: More next_sibling loops here until References head

    return out


def getall(dir_xml, filename):

    # read .xml file
    try:
        with open(os.path.join(dir_xml, filename+'.tei.xml'), 'r') as file: # parse xml as txt
            content = file.read()
    except FileNotFoundError:
        print(("The .xml directory does not exist"))
    soup = bs(content, 'xml') # parse xml

    # , errors= 'ignore'

    # extract every <p>  -- all paragraphs
    out = []
    for plabel in soup.select('p'):
        out.append(plabel.get_text().lower())

    return out



# ================================================================================================
# for .xlsx files
# ================================================================================================

def build_labels_dataframe(dir_xlsx) -> pd.DataFrame:
    '''

    converts a csv file of labels to a dataframe,

    index       StudentID            ArgumentLevel       ReasoningLevel
        0       GS_xxx_Redacted      xyz                 xyz


    input --
        dir_xlsx: directory to xlsx file

    output --
        df_labels: pd dataframe of cleaned xlsx
    '''

    df_labels = pd.read_excel(dir_xlsx)

    # format items in StudentID
    for id in df_labels['StudentID']:
        if not id.startswith('GS_') and not id.endswith('_Redacted'):
            df_labels = df_labels.replace(id, 'GS_'+id+'_Redacted')

        elif not id.endswith('_Redacted'):
            df_labels = df_labels.replace(id, id+'_Redacted')

    # lowercase labels
    for i in df_labels['ArgumentLevel']:
        df_labels = df_labels.replace(i, i.lower())
    for i in df_labels['ReasoningLevel']:
        df_labels = df_labels.replace(i, i.lower())

    return df_labels

#===================================================================================================================================================================================================
  #PICKLE FILES
#===================================================================================================================================================================================================
'''
how to use:
--> the machine learning algorithm could collect your results into a data frame
--> pickle data file to Pickledfiles using function: pickle_save_file
--> unpickle the data file using function: pickle_load_file

'''

def save_as_pickle_file(dataframe, filename, dir_name):
    '''
    dataframe: pandas dataframe
    filename: string name
    '''
    format = 'pkl'
    #dir_name = r'C:\Users\EfiaA\OneDrive - Imperial College London\Imperial academic work\University life\Y4\MSci project\Project_Coding\nlp-physicseducation\Pickledfiles'
    path1 = os.path.join(dir_name,filename + "." + format)
    dataframe.to_pickle(path1)
    return

def load_pickle_file_to_df(filename, dir_name):
    # remeber to add the r'directorryyyy........'
    '''
    filename: string file name
    '''
    format = 'pkl'
    #dir_name = r'C:\Users\EfiaA\OneDrive - Imperial College London\Imperial academic work\University life\Y4\MSci project\Project_Coding\nlp-physicseducation\Pickledfiles'
    path1 = os.path.join(dir_name,filename + "." + format)
    unpickled_dataframe = pd.read_pickle(path1)
    return unpickled_dataframe


#%%
#heatmap
import seaborn as sns
import pandas as pd

x_label = ["EXP", "BAL", "THE", "NONE"]
Y_label = ["Superficial","Extended", "Deep", "Prediction", "Expert"]
array   = [[4,24,12,2],[1,14,9,1],[3,11,5,0],[1,0,0,0],[0,5,3,1]]

#create this matrix into a pandas dataframe
df = pd.DataFrame(array, columns = x_label, index = Y_label)
sns.heatmap(df, square = True, annot= True, linewidths=3)
