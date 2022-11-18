from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd

from tools import utils, formats

# ===================================================================================================================================
dir_xml = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/xml'
dir_txt = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
dir_csv = 'outputs/labels_cleaned.csv'

#dir_xml = '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4/MSci project/Project_Coding/anonymised_reports/anonymised_reports/year_1_2017/cycle_1/xml'
#dir_txt = '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4/MSci project/Project_Coding/anonymised_reports/anonymised_reports/year_1_2017/cycle_1/txt
# ===================================================================================================================================

# iterate over xml files in a folder, and
# map a xml file to a txt file

counter = 0
for file in os.listdir(dir_xml):
    if file.startswith('GS_') and file.endswith('.tei.xml'):
        counter += 1
        filename = file.rsplit('.', maxsplit=2)[0]
        # print(filename)
        utils.xml_to_txt(dir_xml, dir_txt, filename, alltext=True, processtext = True)  # extract all paragraphs
print('Total number of files:', counter)

# -- Get files ---
df_files = utils.build_files_dataframe(dir_txt, 'GS_', '.txt')
# -- Get labels ---
df_labels = utils.build_labels_dataframe('data/labels.xlsx')
# -- Merge dataframes --
df = pd.merge(df_files, df_labels, left_on='StudentID', right_on='StudentID')      # merged dataframe: StudentID, Content, ArgumentLevel, ReasoningLevel
df.to_csv('outputs/labels_cleaned.csv', encoding='utf-8')


'''
# Counts bar plots
df = pd.read_csv(dir_csv, encoding='utf-8')

dummy_dict = df['ArgumentLevel'].value_counts().to_dict()
ArgumentLevel = list(dummy_dict.keys())
counts = list(dummy_dict.values())
formats.bar_plot(ArgumentLevel, counts, xlabel = 'Argument Levels', ylabel = 'Counts', filepath = 'outputs/counts_vs_ArgumentLevel.png')

dummy_dict = df['ReasoningLevel'].value_counts().to_dict()
ReasoningLevel = list(dummy_dict.keys())
counts = list(dummy_dict.values())
formats.bar_plot(ReasoningLevel, counts, xlabel = 'Reasoning Levels', ylabel = 'Counts', filepath = 'outputs/counts_vs_ReasoningLevel.png')
'''
