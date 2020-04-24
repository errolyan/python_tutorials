# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-07-25  00:09
'''

from queue import Queue
import threading
import time

def job(lis,q):
    lis_new = []
    for i in range(len(lis)):
        print(lis[i])
        lis_new.append(lis[i]**2)
    q.put(lis_new)

def multithreading():

    q = Queue()
    threads = []
    data = [[1,2],[3,4],[1,6],[4,5],[7,9],[8,9],[88,1,0,88],[2,4,],[1,3],[11,4,9999]]

    for i in range(10):
        t = threading.Thread(target=job,args=(data[i],q))
        t.start()
        threads.append(t)

    results = []

    for _ in range(10):
        results += q.get()

    for one_thread in threads:
        one_thread.join()


    print(results)





if __name__ == "__main__":
    multithreading()
