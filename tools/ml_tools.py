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
    countvec    = TfidfVectorizer()
    dtm    = countvec.fit_transform(corpus)

    corpus_wordvec_names = countvec.get_feature_names()
    corpus_wordvec_counts  = dtm.toarray()

    return corpus_wordvec_names, corpus_wordvec_counts

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
