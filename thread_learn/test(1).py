
import threading
from queue import Queue

def f(q):

  num = 10000
  print("Wake up....")
  print("Finish put....1")
  q.put('X' * num)
  print("Finish put....2")

def threading_test():
  queue = Queue()
  # p = Process(target=f, args=(queue,))
  p = threading.Thread(target=f, args=(queue,))
  p.start()
  #
  print("Start to sleep...")
  print("Wake up....")
  p.join()
  aaa = queue.get()
  print(aaa)

#threading_test()

from multiprocessing import Process, Queue
import time

def multiprocessing_test():
  queue = Queue()
  p = Process(target=f, args=(queue,))
  p.start()

  print("Start to sleep...")

  aaa = queue.get()
  p.join()
  print(aaa)

multiprocessing_test()  # 注意数据放进队列需要及时取出来，否则大的数据就会有问题。解决办法就是在join之前取出来。