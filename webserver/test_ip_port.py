# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
'@Describe:  http://0.0.0.0:8080/register     params {'id '：1,'name':'yan'}
'@Evn     :
@Date    :  2019-07-25  14:27
'''

from flask import Flask, request

app = Flask(__name__)

def print_(name):
    '''定义自己的功能'''
    print("welcome %s to here"%name)

@app.route('/register', methods=['POST'])
def register():
    print (request.get_data()       )                 # 获取原始数据
    print (request.data  )                               # 同上
    print (request.headers   )                           # 头部信息
    print (request.form       )                          # 所有表单内容
    # print (request.form['name']  )                       # 获取name参数
    print (request.form.get('name',default='yan')   )                  # 如上
    name = request.form.get('name', default='yan')
    print_(name)
    #print (request.form.getlist('name') )             # 获取name的列表
    print (request.form.get('nickname', default='fool') )# 参数无则返回默认值
    return 'welcome'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)