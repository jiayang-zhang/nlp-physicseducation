'''
dump all temporary test code here
'''

from tools import utils

# # test preprocess function
# with open('testfiles/test01.txt', 'r') as f:
#     content = f.read()
# content = utils.preprocess(content)
# print(content)



#  empty xml file
utils.xml_to_txt('/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/xml', '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt', 'GS_KGL587_Redacted', alltext = True, processtext = True)
