# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  @Email:2681506@gmail.com   
@Date:  2019-06-01  01:43
@File：str_.py
@Describe：字符串操作
@Evn:
'''
from io import StringIO
import re

class str_base(object):
    '''字符串处理'''
    def __init__(self,strs):
        self.strs = strs

    def reverse_str(self,):
        '''
        字符串倒序
        :param strs:
        :return:
        '''
        return self.strs[::-1]

    def re_match(self,re_strs):
        '''
        正则表达式的match方法,在单词级别中查找
        :param strs:
        :return:
        '''
        result = re.match(re_strs,self.strs)
        if result is not None:result = result.group()
        return result

    def re_search(self,re_strs):
        '''
        在一个字符串中查找模式
        :param re_strs:
        :return:
        '''
        result = re.search(re_strs,self.strs)
        if result is not None: result = result.group()
        return result

    def re_findall(self,re_strs):
        '''
        查找每一次出现的位置
        :param re_strs: 查找模式
        :return: result 查找结果
        '''
        result = re.findall(re_strs,self.strs,re.I)
        return result

    def re_split(self,re_strs):
        '''
        限定模式上使用split分割字符串
        :param re_strs:
        :return:
        '''
        result = re.split(re_strs,self.strs)
        return result

def main():
    strs = 'This and that'

    newres = str_base(strs)
    print(newres.reverse_str())

    re_strs = 'I'
    print(newres.re_match(re_strs))

    # 匹配单字符
    re_strs = "3\.14"
    print('单字符',newres.re_match(re_strs))

    re_strs = "on"
    print(newres.re_search(re_strs))

    #匹配多个字符串
    re_strs = 'bet|bat|bit'
    print(newres.re_match(re_strs))

    # 查找
    re_strs = r'(th\w+) and (th\w+)'
    print(newres.re_findall(re_strs))

    # 分割
    re_strs = " "
    print(newres.re_split(re_strs))

    sentence = '你丫是傻叉吗? 我操你大爷的. Fuck you.'
    purified = re.sub('[操肏艹]|fuck|shit|傻[比屄逼叉缺吊屌]|煞笔','*', sentence, flags=re.IGNORECASE)
    print(purified)

if __name__=="__main__":
    main()
