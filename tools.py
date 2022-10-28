from bs4 import BeautifulSoup as bs
import os

dir = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/xml'
filename = 'GS_BKZ271_Redacted'

# read .xml file as txt
with open(os.path.join(dir, filename+'.tei.xml'), 'r') as file:
    content = file.read()
#
# parse .xml
soup = bs(content, 'xml')

# display heads
out = []
for head in soup.select('head'):
    if 'result' in head.get_text().lower(): # check lowercase <head>, works with "RESULTS AND"
        s = head.next_sibling
        out.append(s.get_text(' ', strip=True))
        # TO DO: More next_sibling loops here until References head

print(out)

# TO DO: create new txt file
# with open('dir, filename+'.txt', 'w') as file:
    # file.write()
