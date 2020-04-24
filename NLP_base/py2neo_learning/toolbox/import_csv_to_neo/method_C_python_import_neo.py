# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  python 代码导入 csv
@Evn     :  
@Date    :  2019-08-14  08:55
'''
import pandas as pd
from py2neo import Node, Relationship, walk, Graph
from properties_util import Properties


class Import_Node():
    '''
    节点上传类
    '''
    def __init__(self,node_csv_path):

        self.node_csv_path=node_csv_path

    def read_csv(self):
        '''
        读取csv文件
        :return: data,data_head 全数据和头文件
        '''
        data = pd.read_csv(self.node_csv_path)
        head_data = data.columns
        return data,head_data

    def import_node(self):
        data, head_data = self.read_csv()
        print(data,head_data)

if __name__ == "__main__":
    config_path = "../config/csv_make_node.properties"
    node_csv_path = "../../output/Person.csv"
    new_node = Import_Node(node_csv_path)
    new_node.import_node()