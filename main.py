from tools.xmler import *
from pathlib import Path
'''
# uncomment to download them
import nltk
nltk.download('punkt')
nltk.download('stopwords')
'''
# ===================================================================================================================================
dir_xml = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/xml'
dir_txt = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
# ===================================================================================================================================

# iterate over xml files in a folder, and
# map a xml file to a txt file
counter = 0
for file in os.listdir(dir_xml):
    if file.startswith('GS_') and file.endswith('.tei.xml'):
        counter += 1
        filename = file.rsplit('.', maxsplit=2)[0]
        # print(filename)
        xml_to_txt(dir_xml, dir_txt, filename, alltext=True, processtext = False)  # extract all paragraphs
print('Total number of files:', counter)
