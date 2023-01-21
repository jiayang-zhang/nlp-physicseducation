from tools import utils, formats
from pathlib import Path
import os
import pandas as pd

# ===================================================================================================================================
# input
dir_xml = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_2/xml'
dir_txt = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_2/txt'
#dir_xml = '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4/MSci project/Project_Coding/anonymised_reports/anonymised_reports/year_1_2017/cycle_1/xml'
#dir_txt = '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4/MSci project/Project_Coding/anonymised_reports/anonymised_reports/year_1_2017/cycle_1/txt'
csv_in = 'data/labels_y1c2.xlsx'

# output
csv_out = 'outputs/sections/labels_cleaned_y1c2.csv'
# ===================================================================================================================================

'''
# -- unpack xml files --

counter = 0

for file in os.listdir(dir_xml):
    if file.startswith('GS_') and file.endswith('.tei.xml'):

        # Count # of files
        counter += 1

        # Get StudentID
        filename = file.rsplit('.', maxsplit=2)[0]

        # Extract text from reports
        utils.xml_to_txt(dir_xml, dir_txt, filename, alltext=False, processtext = False)

print('Total number of files:', counter)



# -- unpack csv labels and txt files ---

# Build dataframe for texts
df_files = utils.build_files_dataframe(dir_txt, 'GS_', '.txt')

# Built dataframe for labels
df_labels = utils.build_labels_dataframe(csv_in)

# Merge two dataframes
df = pd.merge(df_files, df_labels, left_on='StudentID', right_on='StudentID')      # merged dataframe: StudentID, Content, ArgumentLevel, ReasoningLevel

# Check # of reports
print('# of reports: ',len(df.index))

# Save merged dataframe as csv
df.to_csv(csv_out, encoding='utf-8')
'''

# ===================================================================================================================================
# ===================================================================================================================================
# -- join 2 csv files --

df1 = pd.read_csv('outputs/sections/labels_cleaned_y1c1.csv', encoding='utf-8')
df2 = pd.read_csv('outputs/sections/labels_cleaned_y1c2.csv', encoding='utf-8')
df= df1.append(df2, ignore_index=True)
df.to_csv('outputs/sections/labels_cleaned_y1c1c2.csv', encoding='utf-8')

# ===================================================================================================================================
# ===================================================================================================================================

'''
# -- Count bar plots --
def plot_plots(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    dummy_dict = df['ArgumentLevel'].value_counts().to_dict()
    ArgumentLevel = ['Superficial', 'Extended', 'Deep', 'Prediction', 'Expert', ]
    counts = [dummy_dict[key.lower()] for key in ArgumentLevel]
    print(counts)
    formats.bar_plot(ArgumentLevel, counts, xlabel = 'Argument Levels', ylabel = 'Counts', filepath = 'outputs/counts_vs_ArgumentLevel.png')

    dummy_dict = df['ReasoningLevel'].value_counts().to_dict()
    ReasoningLevel = ['the', 'exp', 'bal', 'none']
    counts = [dummy_dict[key.lower()] for key in ReasoningLevel]
    ReasoningLevel = ['Theoretical', 'Experimental','Balanced', 'None']
    formats.bar_plot(ReasoningLevel, counts, xlabel = 'Reasoning Levels', ylabel = 'Counts', filepath = 'outputs/counts_vs_ReasoningLevel.png')
    return
'''
