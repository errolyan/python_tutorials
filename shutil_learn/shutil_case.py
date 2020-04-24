# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol
@Describe:  shutil是一个比较强大的python的操作文件的包
@Evn     :  pip install shutil
@Date    :  2019-06-26  11:27
'''

import os
import shutil

class shutil_fun(object):
    '''
    learning shutil packages
    '''
    def copy_func(self,source_path,aim_path):
        '''
        将一个文件的内容拷贝的另外一个文件当中
        :param source_path: 源路径
        :param aim_path: 目标路径
        :return:
        '''
        aim_path = shutil.copy(source_path,aim_path)
        return aim_path

    def copyfile_func(self,source_path,aim_path):
        '''
        文件中的内容复制到另外一个文件中
        :param source_path: 源文件路径
        :param aim_path: 目标文件路径
        :return:
        '''
        aim_path = shutil.copyfile(source_path,aim_path)
        return aim_path

    def copytree_func(self,source_dir,aim_dir):
        '''
        复制一个文件夹到另外一个文件下
        :param source_dir:
        :param aim_dir:
        :return:
        '''
        aim_dir = shutil.copytree(source_dir,aim_dir)
        return aim_dir

    def del_dir_func(self,aim_dir):
        '''
        删除真个文件夹
        :param aim_dir: 文件夹路径
        :return:
        '''
        if os.path.exists(aim_dir):
            shutil.rmtree(aim_dir)

    def move_func(self,source_path,aim_path):
        '''
        移动文件
        :param source_path: 源文件路径
        :param aim_path: 目标文件路径
        :return:
        '''
        shutil.move(source_path,aim_path)


def main():
    source_path = "../datasets/test"
    source_dir = "../datasets/"
    aim_path1 = "../output/new_dir"
    aim_path2 = "../output/test"

    new_shutil = shutil_fun()
    # aim_path = new_shutil.copy_func(source_path,aim_path)

    # 复制文件内容到另外一个文件
    # aim_path = new_shutil.copyfile_func(source_path,aim_path2)

    #复制文件夹路径
    # aim_dir = new_shutil.copytree_func(source_dir,aim_path1)

    # 删除文件夹路径
    # new_shutil.del_dir_func(aim_path1)
    new_shutil.move_func(source_path,aim_path2)


if __name__=="__main__":
    main()