## -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol
@Email:2681506@gmail.com
@Date:  2019-05-28  15:48
@File：webserver.py
@Describe: webserver
'''
import json
from flask import Flask
from flask import request
from flask import redirect
from flask import jsonify

# 传递根目录
class WebServer:
    '''
    导入Flask框架，这个框架可以快捷地实现了一个WSGI应用
    '''

    def __init__(self, properties):
        self.port = properties['port']
        self.host = None
        if 'host' in properties.keys():
            self.host = properties['host']
        self.app = Flask('predict service')
        self.properties = properties

        @self.app.route('/',methods=['GET','POST'])
        def predict():
            param_dict = {}
            if request.method == 'POST':
                param = request.json
                print(param)

            else:

                for k, v in request.args.items():
                    param_dict[k] = v
                print(param_dict)

            ret_info = str(11)

            return ret_info

    def start(self):
        '''开始服务 '''
        if self.host != None:
            self.app.run(host=self.host, port=self.port, debug=True, use_reloader=False)
        else:
            self.app.run(port=self.port, debug=True, use_reloader=False)




def main():
    '''
    webserve :实例   http://127.0.0.1:8888/?title=江苏公司千里眼平台数据自动上传总部&docx_path=/Users/yanerrol/Desktop/Text_Similarity/test_input/test.docx&csv_path=/Users/yanerrol/Desktop/Text_Similarity/data/test.csv
    :return:
    '''
    #args = parser_args()
    properties = {"host":"127.0.0.1",'port':"8888"}
    server = WebServer(properties)
    server.start()



if __name__ == "__main__":
    main()