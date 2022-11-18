from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

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

def tf_idf(corpus):
    # performs tf_idf on the dataframe
    v    = TfidfVectorizer()
    x    = v.fit_transform(corpus)
    s_m  = x.toarray()
    return s_m

# ================================================================================================
# supervised classifion
# ================================================================================================

# -- Classification: logistic regression ---
def logistic_regression(X_train, y_train, X_test):

    # Create an instance of LogisticRegression classifier
    lr = LogisticRegression(random_state=0)

    # Fit the model
    lr.fit(X_train, y_train)
    y_test_predict = lr.predict(X_test)

    return  y_test_predict

# ================================================================================================
# evaluation
# ================================================================================================

def sanity_check(X_test, y_test, y_test_predict, printWrong=True):

    # accuracy score
    accuracy = metrics.accuracy_score(y_test, y_test_predict)

    if printWrong:
        flag = True
        for i in range(len(X_test)):
            if y_test[i] != y_test_predict[i]:
                print('wrong prediction:', y_test[i], y_test_predict[i])
                flag = False
        print(flag)

        print('Manual labels:', y_test,'\n','Predicted labels:',y_test_predict)

        print("LogisticRegression Accuracy %.3f" %accuracy)

    return accuracy
