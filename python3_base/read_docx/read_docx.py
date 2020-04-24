# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  
@Email:2681506@gmail.com   
@Date:  2019-05-23  15:10
@File：read_docx.py
@Describe: 读写docx文件
'''

import docx

def read_docx(post_path):
    '''
    read and write docx
    :param post_path:
    :return: title，text_input
    '''
    document = docx.opendocx(post_path)
    paratextlist = docx.getdocumenttext(document)
    print(paratextlist)

    # return title,text_input

def main():
    path = "./data/test.docx"
if __name__ == "__main__":
    main()