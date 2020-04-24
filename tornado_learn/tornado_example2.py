# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： tornado_example2
   Description :  多进程
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/30  14:31
-------------------------------------------------
   Change Activity:
                  2019/12/30  14:31:
-------------------------------------------------
'''
__author__ = 'yanerrol'
# coding:utf-8

import tornado.web
import tornado.ioloop
import tornado.httpserver

class IndexHandler(tornado.web.RequestHandler):
    """主路由处理类"""
    def get(self):
        """对应http的get请求方式"""
        self.write("Hello Itcast!")

if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/", IndexHandler),
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    # -----------修改----------------
    http_server.bind(8000)
    http_server.start(0)
    # ------------------------------
    tornado.ioloop.IOLoop.current().start()