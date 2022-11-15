'''
dump all temporary test code here
'''

from tools import utils

# test preprocess function
with open('testfiles/test01.txt', 'r') as f:
    content = f.read()
content = utils.preprocess(content)
print(content)
