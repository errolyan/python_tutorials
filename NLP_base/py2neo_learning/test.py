# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  neo4j 基础教程和py2neoAPI
@Evn     :  pip install py2neo
@Date    :  2019-08-02  09:40
'''

from py2neo import Node, Relationship

a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
r = Relationship(a, 'KNOWS', b)
print("a, b, r:",a, b, r)

a['age'] = 20
b['age'] = 21
r['time'] = '2017/08/31'
print("a, b, r:",a, b, r)

a.setdefault('loaction','BeiJing')
print("a:",a)

# 批量更新属性值
data = {"name":u"Yan erle","age:":22}
a.update(data)
print("a",a)

# 交集
s1 = a | b | r
s2 = a | b
print(s1 & s2)

a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
r = Relationship(a, 'KNOWS', b)
s = a | b | r
# 属性或者节点
s = a | b | r
print("s.keys()",s.keys())

# print("s.nodes()",s.nodes())
# print(s.relationships())
print(s.types())