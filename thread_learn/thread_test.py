# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-07-24  23:54
'''

from queue import Queue
import threading
import time

def thread_job():
    print("thread 1 Start \n")
    for i in range(10):
        time.sleep(0.1)
    print("thread 1 finish \n")

def thread2_job():
    print("thread 1 Start \n")
    for i in range(10):
        time.sleep(1)
    print("thread 1 finish \n")

def f(q):
  num = 100000
  q.put('X' * num)
  print("Finish put....")

def main():
    added_thread1 = threading.Thread(target = thread_job,name = "T1")
    added_thread2 = threading.Thread(target=f, name="T2")
    added_thread1.start()
    added_thread2.start()
    #added_thread1.join()
    added_thread2.join()
    print("All threadings is done")

if __name__ == "__main__":
    main()