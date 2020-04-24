# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-1723:55
Describe:二分法
'''

import time

def binary_search(inputlist,item):
    first = 0
    last = len(inputlist) - 1
    while first <= last:
        midlist = int((first + last)/2)
        if inputlist == item:
            return True
        elif item < inputlist[midlist]:
            last = midlist - 1
        else:
            first = midlist +1

def main():
    testli = [99,102,104,106,107,199]
    print(binary_search(testli,102))
    print(binary_search(testli,107))


if __name__ == "__main__":
    main()