from tools import *
'''
# uncomment to download them
import nltk
nltk.download('punkt')
nltk.download('stopwords')
'''
# ================================================================================================

dir_xml = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/xml'
dir_txt = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
filenames = ['test01', 'test02']
# ================================================================================================

# map .xml to .txt
# for filename in filenames:
    # xml_to_txt(dir_xml, dir_txt, filename)


# create totaltext.txt
# total = []
# for filename in filenames:
#     with open(os.path.join(dir_txt, filename+'.txt'), 'r') as file:
#         content = file.read()
#     total.append('  ,  ')
#     total.append(content)
# with open(os.path.join(dir_txt, 'totaltext.txt'), 'w') as file:
#     file.write(' '.join(total))



# convert files to tokens
'''
... each file --> tokens (saved to pandas)
... totaltext.txt --> tokens (saved to pandas)
https://thatascience.com/learn-machine-learning/bag-of-words/
...
....
'''
for filename in filenames:
    with open(os.path.join(dir_txt, filename+'.txt'), 'r') as file:
        content = file.read()
    tokens = tokeniser(content)

print(sorted(tokens))

'''
... BoW
'''
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(tokens)
# print(sorted(vectorizer.get_feature_names()))
#
# print(X.toarray())
