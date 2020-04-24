# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''

import re

def is_number(num):
  pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
  result = pattern.match(num)
  if result:
    return True
  else:
    return False


def numberChange(path):
    with open(path,'r') as f:
        data = f.read()
        number = re.compile('([一二三四五六七八九零十百千万亿壹贰叁肆伍陆柒捌玖拾角分佰仟亿]+|[0-9]+[,]*[0-9]+.[0-9])')
        pattern = re.compile(number)
        all = pattern.findall(data)
        for i in all:
            print(i)
            if  is_number(i):
                print('True')
            else:
                print('False')




if __name__ == '__main__':
    path = './mytest.txt'
    numberChange(path)