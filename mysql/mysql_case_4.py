# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol <https://blog.csdn.net/weixin_32393347,https://github.com/errolyan>
@Describe:  
@Evn     :  
@Date    :  2019-06-24  14:55
'''

import pymysql
from pymysql.cursors import DictCursor


def main():
    # con = pymysql.connect(host='0.0.0.0', port=3306,
    #                       database='hrs', charset='utf8',
    #                       user='root', password='yel219')
    con = pymysql.connect(host='182.92.158.97', port=3306, database='gegeda', charset='utf8', user='root',
                          password='253c0cgeg41')

    try:
        with con.cursor(cursor=DictCursor) as cursor:
            cursor.execute('select audio_status as audio_status, audio as audio, id as id from gg_video')
            results = cursor.fetchall()
            print(results)

    finally:
        con.close()


if __name__ == '__main__':
    main()
