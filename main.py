from tools import *

# ================================================================================================

dir_xml = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/xml'
dir_txt = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
filename = 'GS_BKZ271_Redacted'
# ================================================================================================


xml_to_txt(dir_xml, dir_txt, filename)
tokens = tokeniser(dir_txt,filename)
print(tokens)
