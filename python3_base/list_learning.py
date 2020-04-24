# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-10  23:36
'''

from collections import deque
'''
解决大的列表删除和检索比较繁琐，删除效率低，
'''
q = deque(['a',"b","c"])
q.append("x")
q.appendleft("y")

print(q)

from collections import ChainMap
import os, argparse

'''
什么时候使用ChainMap最合适？举个例子：应用程序往往
都需要传入参数，参数可以通过命令行传入，可以通过环境
变量传入，还可以有默认参数。我们可以用ChainMap实现参
数的优先级查找，即先查命令行参数，如果没有传入，再查环
境变量，如果没有，就使用默认参数。
'''
# 构造缺省参数:
defaults = {
    'color': 'red',
    'user': 'guest'
}

# 构造命令行参数:
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user')
parser.add_argument('-c', '--color')
namespace = parser.parse_args()
command_line_args = { k: v for k, v in vars(namespace).items() if v }

# 组合成ChainMap:
combined = ChainMap(command_line_args, os.environ, defaults)

# 打印参数:
print('color=%s' % combined['color'])
print('user=%s' % combined['user'])