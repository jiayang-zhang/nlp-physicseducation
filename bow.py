from tools import ml_tools, utils
import os
import pandas as pd
pd.set_option('max_colu', 10)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# =======================================================================================================================================
dir_txtfldr = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
# =======================================================================================================================================

# -- Get files ---
df_files = utils.build_files_dataframe(dir_txtfldr, 'GS_', '.txt')
# -- Get labels ---
df_labels = utils.build_labels_dataframe('data/labels.xlsx')
# -- Merge dataframes --
df = pd.merge(df_files, df_labels, left_on='StudentID', right_on='StudentID')      # merged dataframe: StudentID, Content, ArgumentLevel, ReasoningLevel


# -- Feature extraction: Bag of Words ---
wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())


# -- Classification: logistic regression ---

iterations = 10
sum = 0
for i in range(iterations):
    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(wordvec_counts, df['ArgumentLevel'].tolist(), train_size = 0.8)
    # print(len(X_train))
    # print(len(y_train))

    # Create an instance of LogisticRegression classifier
    lr = LogisticRegression(random_state=0)
    # Fit the model
    lr.fit(X_train, y_train)

    # Sanity check - training data
    # ml_tools.sanity_check(lr.predict, X_train, y_train, printWrong=False)

    # Prediction - test data
    accuracy_score = ml_tools.sanity_check(lr.predict, X_test, y_test, printWrong=False)
    sum += accuracy_score

print('average accuracy score:', sum/iterations)
