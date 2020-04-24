# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol <https://blog.csdn.net/weixin_32393347,https://github.com/errolyan>
@Describe:  
@Evn     :  pip install pymysql
@Date    :  2019-06-24  14:51
'''

import pymysql


def main():
    no = int(input('编号: '))
    con = pymysql.connect(host='localhost', port=3306,
                          database='hrs', charset='utf8',
                          user='root', password='yel219',
                          autocommit=True)
    try:
        with con.cursor() as cursor:
            result = cursor.execute(
                'delete from tb_dept where dno=%s',
                (no, )
            )
        if result == 1:
            print('删除成功!')

    finally:
        con.close()


if __name__ == '__main__':
    main()