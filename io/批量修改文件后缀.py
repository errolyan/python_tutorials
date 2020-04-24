# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： 批量修改文件后缀
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/31  14:06
-------------------------------------------------
   Change Activity:
          2020/3/31 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
    description='工作目录中文件后缀名修改')
    parser.add_argument('work_dir', metavar='WORK_DIR', type=str,nargs=1,help='修改后缀名的文件目录')
    parser.add_argument('old_ext', metavar='OLD_EXT',type=str, nargs=1, help='原来的后缀')
    parser.add_argument('new_ext', metavar='NEW_EXT',type=str, nargs=1, help='新的后缀')
    return parser

def batch_rename(work_dir, old_ext, new_ext):
    """
   传递当前目录，原来后缀名，新的后缀名后，批量重命名后缀"""
    for filename in os.listdir(work_dir): # 获取得到文件后缀
        split_file = os.path.splitext(filename)
        file_ext = split_file[1]
        # 定位后缀名为 old_ext 的文件
        if old_ext == file_ext:
            # 修改后文件的完整名称
            newfile = split_file[0] + new_ext # 实现重命名操作
            os.rename(os.path.join(work_dir, filename),os.path.join(work_dir, newfile))
    print("完成重命名")
    print(os.listdir(work_dir))

def main():
    """
    main 函数
    """
    # 命令行参数
    parser = get_parser()
    args = vars(parser.parse_args()) # 从命令行参数中依次解析出参数
    work_dir = args['work_dir'][0]
    old_ext = args['old_ext'][0]
    if old_ext[0] != '.':
        old_ext = '.' + old_ext
    new_ext = args['new_ext'][0]
    if new_ext[0] != '.':
        new_ext = '.' + new_ext
    batch_rename(work_dir, old_ext, new_ext)

if __name__ == '__main__':
    main()