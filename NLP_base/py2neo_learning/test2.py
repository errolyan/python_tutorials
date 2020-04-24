# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol
@Describe:
@Evn     :
@Date    :  2019-08-02  13:59
'''

from py2neo import Node, Relationship, walk, Graph
print(__doc__)

a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
c = Node('Person', name='Mike')
ab = Relationship(a, "KNOWS", b)
ac = Relationship(a, "KNOWS", c)
w = ab + Relationship(b, "LIKES", c) + ac
for item in walk(w):
    print(item)

a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
r = Relationship(a, 'KNOWS', b)
s = a | b | r
# 存储数据库
graph = Graph(host="localhost",password='yel219')
graph.create(w)

# 查询语句
node = graph.match(nodes='Person')
print(node)

relationship = graph.match_one(r_type='KNOWS')
print(relationship)

# update data
# a = Node('Person',name='Alice')
# node = graph.match_one(nodes='Person',r_type='KNOWS')
# node['age']=100
# graph.push(node)
# print(graph.match_one(nodes = 'Person',r_type='KNOWS'))

# delete node
# node = graph.match_one(nodes='Person')
# relationship = graph.match_one(r_type='KNOWS')
# graph.delete(relationship)
# graph.delete(node)
