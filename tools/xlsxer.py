'''
tools for processing xlsx files
'''

import pandas as pd



def build_labels_dataframe(dir_xlsx):
    '''

    converts a csv file of labels to a dataframe,

    index       StudentID            ArgumentLevel       ReasoningLevel
        0       GS_xxx_Redacted      xyz                 xyz


    input --
        dir_xlsx: directory to xlsx file

    output --
        df_labels: pd dataframe of cleaned xlsx
    '''

    df_labels = pd.read_excel(dir_xlsx)

    # format items in StudentID
    for id in df_labels['StudentID']:
        if not id.startswith('GS_') and not id.endswith('_Redacted'):
            df_labels = df_labels.replace(id, 'GS_'+id+'_Redacted')

        elif not id.endswith('_Redacted'):
            df_labels = df_labels.replace(id, id+'_Redacted')

    # lowercase labels
    for i in df_labels['ArgumentLevel']:
        df_labels = df_labels.replace(i, i.lower())
    for i in df_labels['ReasoningLevel']:
        df_labels = df_labels.replace(i, i.lower())

    return df_labels
