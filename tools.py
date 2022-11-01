from bs4 import BeautifulSoup as bs
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import numpy as np



def xml_to_txt(dir_xml, dir_txt, filename):
    """
    saves a single .xml file (from RESULTS up to REREFENCES) to a .txt file
    """

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
            out.append(s.get_text(' ', strip=True)) #string
            # TO DO: More next_sibling loops here until References head
    # print(out)

    # write output to .txt file
    try:
        with open(os.path.join(dir_txt, filename+'.txt'), 'w') as file:
            file.write(' '.join(out))
    except FileNotFoundError:
        print("The .txt directory does not exist")

    return




def tokeniser(text):
    '''
    splits string into tokens
        (and removes punctuation, stopword tokens)

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
    # print('Punctuation')
    # print(tokens)

    # 3. remove stop words tokens
    tokens = list(filter( (lambda t: t not in stopwords.words('english')), tokens ))
    # print('stopwords')
    # print(tokens)

    # TO DO: consider words like 'eighty-seven'
    return tokens
