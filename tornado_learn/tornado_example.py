# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： tornado_example
   Description :  Tornado走的是少而精的方向，注重的是性能优越，它最出名的是异步非阻塞的设计方式。HTTP服务器、异步编程、WebSockets
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/30  14:27
-------------------------------------------------
   Change Activity:
                  2019/12/30  14:27:
-------------------------------------------------
'''
__author__ = 'yanerrol'

'''单进程'''

import tornado.web
import tornado.ioloop

class IndexHandler(tornado.web.RequestHandler):
    """主路由处理类"""
    def get(self):
        """对应http的get请求方式"""
        self.write("Hello Itcast!")

if __name__ == "__main__":
    app = tornado.web.Application([(r"/", IndexHandler),])
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()



