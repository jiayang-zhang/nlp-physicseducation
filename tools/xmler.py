'''
tools for processing xml files, and txt files
'''

from bs4 import BeautifulSoup as bs
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np


# ========================================================================================
#       Functions for .txt files
# ========================================================================================

def build_files_dataframe(dir_txtfldr, str_start, str_end):
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

    # token back to text
    text = ' '.join(str(x) for x in tokens)

    # TO DO: consider words like 'eighty-seven'
    return text  # change to text





# ========================================================================================
#       Functions for .xml files
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
    if alltext == True:
        out = getall(dir_xml, filename)
    else:
        out = getparagraphs(dir_xml, filename)


    # write output to .txt file
    try:
        with open(os.path.join(dir_txt, filename+'.txt'), 'w') as file:
            content = ' '.join(out)
            if processtext == True:
                file.write(preprocess(content))
            else:
                file.write(content)
    except FileNotFoundError:
        print("The .txt directory does not exist")

    return


def getparagraphs(dir_xml, filename):
    '''
    Not ready yet...
    '''
    # read .xml file
    try:
        with open(os.path.join(dir_xml, filename+'.tei.xml'), 'r') as file: # parse xml as txt
            content = file.read()
    except FileNotFoundError:
        print(("The .xml directory does not exist"))
    soup = bs(content, 'xml') # parse xml


    # extract <head> and <p>
    out = []
    for head in soup.select('head'):
        if 'result' in head.get_text().lower(): # check lowercase <head>, works with "RESULTS AND"
            s = head.next_sibling
            if s == None:
                print('No text found...')
                print('error filename:',filename)
            else:
                out.append(s.get_text(' ', strip=True)) #string
            # TO DO: More next_sibling loops here until References head
    # print(out)

    return out


def getall(dir_xml, filename):

    # read .xml file
    try:
        with open(os.path.join(dir_xml, filename+'.tei.xml'), 'r') as file: # parse xml as txt
            content = file.read()
    except FileNotFoundError:
        print(("The .xml directory does not exist"))
    soup = bs(content, 'xml') # parse xml

    # extract every <p>  -- all paragraphs
    out = []
    for plabel in soup.select('p'):
        out.append(plabel.get_text().lower())

    return out
