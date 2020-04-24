# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  @Email:2681506@gmail.com   
@Date:  2019-05-31  13:57
@File：re_class.py
@Describe:正则表达式类
@Evn:
'''

import re

class re_cls(object):

    def __init__(self,strs):
        '''
        对象属性
        :param strs:
        '''
        self.strs = strs

    def is_alpha(self):
        '''
        判断是否识字母
        :return:
        '''
        res = self.strs.isalpha()
        print("res", res, type(res), id(res))
        res = self.strs.isalnum()
        print("res",res,type(res),id(res))

        return res


    def re_find(self,find_str):
        '''
        正则搜索
        :param find_str:目标字符串
        :return: 返回匹配的对象 返回一个列表
        '''
        res = re.findall(find_str,self.strs)
        print("res",res,type(res),id(res))
        print(re.DEBUG)
        return res

    def re_rearch(self,find_str):
        '''
        搜索字符串
        :param find_str: 目标字符串
        :return:返回匹配的对象
        '''
        res = re.search(find_str,self.strs)
        print("res", res, type(res), id(res))
        return res

    def re_match(self,find_str):
        '''
        匹配字符串
        :param find_str: 目标字符串
        :return: 匹配的对象
        '''
        res = re.match(find_str,self.strs)
        print("res", res, type(res), id(res))
        return res

    def re_split(self,find_str):
        '''
        按照find_str分开string
        :param find_str: 拆分的符号
        :return:
        '''
        res = re.split(find_str, self.strs)
        print("res", res, type(res), id(res))
        return

    def re_sub(self,find_str,repl):
        '''
        替换对应的字符
        :param find_str: 目标字符串
        :param repl: 替换的对象
        :return:
        '''
        res = re.sub(find_str, repl, self.strs, count=0, flags=0)
        print("res", res, type(res), id(res))
        return res

def main():
    strs = "我是中中国国人国国，我在练习r e，RE . test，200%，3000元   9000万"
    newstr = re_cls(strs)
    find_str = "200%"
    repl = "huihui"
    res_isa = newstr.is_alpha()
    res_list = newstr.re_find(find_str)
    res_weig = newstr.re_rearch(find_str)
    res_mat = newstr.re_match(find_str)
    res_split = newstr.re_split(find_str)
    res_sub = newstr.re_sub(find_str,repl)
    list = ['床前明月光', '疑是地上霜', '举头望明月', '低头思故乡']
    print('-'.join(list))


if __name__ == "__main__":
    main()