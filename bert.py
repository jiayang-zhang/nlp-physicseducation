from tools import ml_tools, utils
import os
import pandas as pd
pd.set_option('max_colu', 10)


from sklearn.model_selection import train_test_split

# =======================================================================================================================================
dir_txtfldr = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
# =======================================================================================================================================

# -- Get files ---
df_files = utils.build_files_dataframe(dir_txtfldr, 'GS_', '.txt')
# -- Get labels ---
df_labels = utils.build_labels_dataframe('data/labels.xlsx')
# -- Merge dataframes --
df = pd.merge(df_files, df_labels, left_on='StudentID', right_on='StudentID')      # merged dataframe: StudentID, Content, ArgumentLevel, ReasoningLevel
