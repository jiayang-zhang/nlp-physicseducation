# -*- coding: utf-8 -*-
import pandas as pd
import os
from bs4 import BeautifulSoup as bs
dir1  = '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4/MSci project/Project_Coding/anonymised_reports/anonymised_reports/year_1_2017/cycle_1/xml'
filename = 'GS_BKZ271_Redacted'

out = []
file_names = []
count = 0
for file in os.listdir(dir1):
    file_names.append(file)
    with open(os.path.join(dir1, file), 'r', encoding = 'utf-8') as f:
        contents = f.read()
        count +=1  # there are 86 files  (unit test checks)
        soup = bs(contents, 'xml')
        element = soup.p # only able to get the first paragrah
        #out.append(element)
        nextSiblings = element.find_next_siblings("p")
        out.append(nextSiblings)
        
        
# storing the lab reports in a pandas datafram
data_table = pd.DataFrame(out, index = file_names)
print(data_table.iloc[1,0])


#%%
# started creating the above piece of code into a class
class DataStorage:
    def __init__(self, paths):
        self.path = paths
    
    def openfiles(self):
        for file in os.listdir(dir1):
            with open(os.path.join(dir1, file), 'r', encoding = 'utf-8') as f:
                contents = f.read()
                soup = bs(contents, 'xml')

x = DataStorage(dir1)

    