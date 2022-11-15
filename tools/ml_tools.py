from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

# ================================================================================================
# feature extraction
# ================================================================================================
def BoW(corpus):
    '''
    performs Bag of Words
    inputs --
        corpus:     a list of strings
                    [report1content... , report2content..., reportNcontent... ]
    returns --
        corpus_wordvec_names:       word vector name list
        corpus_wordvec_counts:      word vector frequncy list
                                    # len(corpus_wordvec_counts) = len(corpus)
    '''
    countvec = CountVectorizer()
    dtm = countvec.fit_transform(corpus)  # document-term matrix

    corpus_wordvec_names = countvec.get_feature_names()     # word vector name list
    corpus_wordvec_counts = dtm.toarray()                   # word vector frequncy list    # len(corpus_wordvec_counts) = len(corpus)

    return corpus_wordvec_names, corpus_wordvec_counts

# ================================================================================================
# evaluation
# ================================================================================================

def sanity_check(model, X_input, y_input, printWrong=True):
    y_predict = model(X_input)

    if printWrong:
        flag = True
        for i in range(len(X_input)):
            if y_input[i] != y_predict[i]:
                print('wrong prediction:', y_input[i], y_predict[i])
                flag = False
        # print(flag)
    # print('Manual labels:', y_input,'\n','Predicted labels:',y_predict)

    print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_input, y_predict))

    return