
# feature extraction imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


#machine learning model imports
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier



#cross validation imports
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#evaluation imports
from sklearn import metrics

# ================================================================================================
# feature extraction
# ================================================================================================
#-------Bag-of-words------------
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


#----------TF-IDF------------------

def tf_idf(corpus):
    # performs tf_idf on the dataframe
    countvec    = TfidfVectorizer()
    dtm    = countvec.fit_transform(corpus)

    corpus_wordvec_names = countvec.get_feature_names()
    corpus_wordvec_counts  = dtm.toarray()

    return corpus_wordvec_names, corpus_wordvec_counts

'''
    v    = TfidfVectorizer()
    x    = v.fit_transform(corpus)
    s_m  = x.toarray()
    return s_m
'''

# ================================================================================================
# supervised classifion
# ================================================================================================

# -- Classification: logistic regression ---
def logistic_regression(X_train, X_test, y_train):

    # Create an instance of LogisticRegression classifier
    lr = LogisticRegression(random_state=0)
    # Fit the model
    lr.fit(X_train, y_train)
    y_test_predict = lr.predict(X_test)
    return  y_test_predict

# -- Classification: Naive Bayes ------------

def naive_bayes(X_train, y_train, X_test, y_test):
    nb = MultinomialNB().fit(X_train, y_train)
    predictions = nb.predict(X_test) # array of predicted labels
    acc_score = metrics.accuracy_score(y_test, predictions) # single accuracy score
    return predictions

# -- Classification: Random forest ----------

def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(max_depth= None, random_state = 0).fit(X_train, y_train)
    predictions = rf.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, predictions) # single accuracy score
    return predictions


# ================================================================================================
# EVALUATION
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

def accuracy_score(y_test, predictions):
    # accuracy scores
    acc_score   = metrics.accuracy_score(y_test, predictions)
    return acc_score

#============================================================================================================
# CROSS-VALIDATION: KFOLD
#============================================================================================================

def kfold(model,x, df_y, n_iterations):
    '''
    returns
    '''
    kf = KFold(n_splits= n_iterations)
    #df_y = df['ReasoningLevel'].tolist()
    results = cross_val_score(model, x, df_y, cv = kf)
    return results
