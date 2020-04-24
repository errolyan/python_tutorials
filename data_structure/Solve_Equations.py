# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-10--23:33
Describe: 解决x + y + z=1000, x^2 + y^2 = z^2解的个数
'''
import time


def solve_equtions_enmu(n):
    # x^2+y^2+z^2=n
    lis1 = []
    li = []
    for x in range(0, n + 1):
        for y in range(0, n + 1):
            for z in range(0, n + 1):
                if x * x + y * y == z * z and x + y + z == n:
                    print("x,y,z: %d,%d,%d" % (x, y, z))
                    li.append(x)
                    li.append(y)
                    li.append(z)
                    lis1.append(li)
    return lis1

# 优化后的算法


def solve_equtions_youhua(n):
    lis1 = []
    li = []
    for x in range(0, n + 1):
        for y in range(0, n + 1):
            z = n - x - y
            if x * x + y * y == z * z:
                print("x,y,z: %d,%d,%d" % (x, y, z))
                li.append(x)
                li.append(y)
                li.append(z)
                lis1.append(li)
    return lis1


def main():
    n = int(input("x + y + z ="))
    start_time_1 = time.time()
    list1 = solve_equtions_enmu(n)
    print(list)
    end_time_1 = time.time()
    print("枚举计算的时间是%f" % (end_time_1 - start_time_1))

    start_time_2 = time.time()
    list1 = solve_equtions_youhua(n)
    print(list)
    end_time_2 = time.time()
    print("算法优化后的计算的时间是%f" % (end_time_2 - start_time_2))

    print("complete! 优化后的算法节省时间：%f" %
          ((end_time_1 - start_time_1) - (end_time_2 - start_time_2)))


if __name__ == "__main__":
    main()
