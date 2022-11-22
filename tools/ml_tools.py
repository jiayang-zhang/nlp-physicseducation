
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
'''
def tf_idf(corpus):
    # performs tf_idf on the dataframe
<<<<<<< HEAD
    countvec    = TfidfVectorizer()
    dtm    = countvec.fit_transform(corpus)

    corpus_wordvec_names = countvec.get_feature_names()
    corpus_wordvec_counts  = dtm.toarray()

    return corpus_wordvec_names, corpus_wordvec_counts
'''

=======
    v    = TfidfVectorizer()
    x    = v.fit_transform(corpus)
    s_m  = x.toarray()
    return s_m

>>>>>>> 1030d7d3160183bb7f9f90d4c91c7f23a6c601d7
# ================================================================================================
# supervised classifion
# ================================================================================================

# -- Classification: logistic regression ---
def logistic_regression(X_train, y_train):
    # Create an instance of LogisticRegression classifier
    lr = LogisticRegression(random_state=0)
    # Fit the model
    lr.fit(X_train, y_train)

    return lr.predict # returns classifier model

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
    return predictions


# ================================================================================================
# EVALUATION
# ================================================================================================

def sanity_check(model, X_input, y_input, printWrong=True):
    y_predict = model(X_input)

    # accuracy score
    accuracy = metrics.accuracy_score(y_input, y_predict)

    if printWrong:
        # flag = True
        # for i in range(len(X_input)):
        #     if y_input[i] != y_predict[i]:
        #         print('wrong prediction:', y_input[i], y_predict[i])
        #         flag = False
        # print(flag)

        # print('Manual labels:', y_input,'\n','Predicted labels:',y_predict)

        print("LogisticRegression Accuracy %.3f" %accuracy)

    return accuracy

<<<<<<< HEAD



def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity

# reference: https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894
=======
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


>>>>>>> 1030d7d3160183bb7f9f90d4c91c7f23a6c601d7
