# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  csv 文件创建节点和关系
@Evn     :  
@Date    :  2019-08-13  08:45
'''

import os
import pandas as pd
import hashlib
import csv
from properties_util import Properties


def get_md5(string):
    """Get md5 according to the string
    """
    byte_string = string.encode("utf-8")
    md5 = hashlib.md5()
    md5.update(byte_string)
    result = md5.hexdigest()
    return result


def read_data(csv_input_path):
    '''
    读取数据
    :param path: file path(.csv or .xlsx)
    :return: data(all_data)、head_data(表头）
    '''
    if os.path.splitext(csv_input_path)[1] == ".csv":
        data = pd.read_csv(csv_input_path)
        head_data = data.columns
    else:
        data = pd.read_excel(csv_input_path)
        head_data = data.columns
    return data, head_data


def del_colum(data,drop_colum):
    '''
    删除要删除的列
    :param data: 完整的数据
    :param drop_colum: 要删除的列的表头，type是列表
    :return: data
    '''
    data = data.drop(columns=drop_colum)
    return data


def remoce_repetition(data):
    '''
    去除重复的行，
    :param data: 删除多余列的数据
    :return: data：去重复后的数据
    '''
    data = data.drop_duplicates()
    return data


def make_node(data,node_properties,node_id,outpathdir):
    '''
    删除多余列和删除重复行的pd数据
    :param data: 输入数据
    :param node_properties: 节点属性
    :param node_id: 节点名字
    :return:
    '''
    node_properties.insert(0,"id")
    node_properties.append(node_id)
    print("node_properties", node_properties)
    outpath = os.path.join(outpathdir, node_id+".csv")
    with open(outpath,"w") as file_import:
        file_import_csv = csv.writer(file_import, delimiter=',')

        file_import_csv.writerow(node_properties)

        node_properties_samples = data[node_properties[1:-1]]
        for index, row in node_properties_samples.iterrows():
            if index == 0:
                continue
            info = []
            for item in node_properties[1:-1]:
                info.append(row[item])

            info_str = str(tuple(info))
            info_id = get_md5(info_str)
            info.insert(0, info_id)

            info.append(node_id)

            file_import_csv.writerow(info)
    print("-done")


def build_node_node_relation(data,node_list,type_relation, outpathdir):
    '''

    :param data: 删除多余列和去重后的信息
    :param node_list: 节点列表
    :param type_relation: 关系
    :param outpathdir: 输出路径
    :return:
    '''

    outpath = os.path.join(outpathdir, type_relation + ".csv")

    with open(outpath, 'w', encoding='utf-8') as file_import:
        file_import_csv = csv.writer(file_import, delimiter=',')

        node_list.insert(0, "relation_id")
        node_list.append(type_relation)
        file_import_csv.writerow(node_list)
        # print("node_list",node_list)

        nodes_relation = data[node_list[1:-1]]
        print("nodes_relation",nodes_relation)

        for index, row in nodes_relation.iterrows():
            if index == 0:
                continue
            info = []
            for item in node_list[1:-1]:
                info.append(row[item])

            info_str = str(tuple(info))
            info_id = get_md5(info_str)
            info.insert(0, info_id)

            info.append(type_relation)

            file_import_csv.writerow(info)
    print("-done")


def csv_import_neo4j(config_path):
    '''
    csv 转化为节点和关系的main函数
    :param config_path: config 路径
    :return:
    '''
    dic_pro = Properties(config_path).get_properties()

    csv_output_path = dic_pro["csv_output_path"]
    if not os.path.exists(csv_output_path):
        os.makedirs(csv_output_path)

    csv_input_path = dic_pro["csv_input_path"]
    if not os.path.isfile(csv_input_path):
        print("not exist data csv！")

    data,head_data = read_data(csv_input_path)
    print(data,"\n",head_data)

    drop_colum = dic_pro["drop_colum"]
    drop_colum = drop_colum.split(",")
    if len(drop_colum) != 0:
        data = del_colum(data, drop_colum)

    data = remoce_repetition(data)
    print(data)

    outpathdir = csv_output_path
    node_id = dic_pro["node_id_1"]
    node_properties = dic_pro["node_properties_1"].split(",")
    make_node(data, node_properties, node_id, outpathdir)

    node_id = dic_pro["node_id_2"]
    node_properties = dic_pro["node_properties_2"].split(",")
    make_node(data, node_properties, node_id, outpathdir)

    node_list = dic_pro["node_list"].split(",")
    type_relation = dic_pro["type_relation"]
    build_node_node_relation(data, node_list, type_relation, outpathdir)

if __name__ == '__main__':
    config_path = "../../config/csv_make_node.properties"
    csv_import_neo4j(config_path)
    # import_path = '../data/import'
    # if not os.path.exists(import_path):
    #     os.makedirs(import_path)