# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： sqlite_learning
   Description :  APP python微数据库
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/2  11:29
-------------------------------------------------
   Change Activity:
                  2020/2/2  11:29:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import sqlite3
import os

# 链接数据库
path = os.getcwd()
files = os.listdir(path)
conn = sqlite3.connect(path+'\db.db')
cur = conn.cursor()

cur.execute('create table if not exists numbers (id integer primary key ,number varchar(20) NOT NULL)')
conn.commit()

# 写数据
i = 0
for file in files:
    if file.split('.')[-1]=='txt':
        with open(file,'r',encoding='utf-8') as f:
            next(f)
            for line in f:
                i += 1
                print("插入第",i,'条数据：')
                cur.execute('insert into numbers values(?,?)', (i, line))
                conn.commit()
                cur.close()
conn.close()
print('数据写入完成！共写入', i,' 条数据')

# 查询数据库中有哪些表
# cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
# Tables=cur.fetchall()
# print(Tables)

# 删除数据库中的某个表
cur.execute("drop table tablename;")

# 查询某一个表的结构
cur.execute("PRaGMA table_info(numbers)")
print(cur.fetchall())

# 查询表中前50条记录
cur.execute("SELECT * from numbers limit 0,50;")
conn.commit
data = cur.fetchall()
print(data)

# 查询表中不重复的记录
cur.execute("SELECT distinct number from numbers;")
data_distinct = cur.fetchall()
b = len(data_distinct)
print('共有 '+ str(b) +' 条不重复记录')
