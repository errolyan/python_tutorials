#! /usr/bin/python
# -*- coding: utf-8 -*-
# __author__ = "errrolyan"
# __Date__: 18-12-10
# __Describe__ = "感知器perceptron 算法Python实现版本，主要实现“异或问题，结构为两层感知器结构，输入层、隐含层、输出层”

import os
import random

class perceptron():
    #使用三层感知器来解决异或问题

    def __init__(self):
        self.data = [[0,0,0],[1,1,1],[0,1,0],[1,0,0]]
        self.rate = 0.05  # 学习率
        self.error = 0.0025 # 终止训练误差

    def datapropare(self):
        dataSet = []
        target = []
        for i in range(len(self.data)):
            dataSet.append(self.data[i][0:2])
            target.append(self.data[i][2])
            i += 1
        for i in range(len(self.data)):
            dataSet[i].append(1)
            i += 1
        return dataSet,target
    def weight(self):
        Wghlist2 =[]
        for i in range(3):
            Wghlist2.append(random.uniform(0,1))
        return Wghlist2

    def accumulation(self,dataSet,Wghlist):
        accumValue = 0
        for i in range(3):
            accumValue += dataSet[i]*Wghlist[i]
        return accumValue

    def Activationfun(self,accumValue): #界越函数
        if accumValue >= 0:
            PredictValue = 1
        else:
            PredictValue = 0
        return PredictValue


    def deffiValue(self,predictarget,target):
        errvalue = target - predictarget
        return errvalue

    def adjustWght(self,errvalue,Wghlisti,dataseti):
        Wghlisti1 = Wghlisti + self.rate*errvalue*dataseti
        return Wghlisti1

    def perceptron(self):
        dataSet, target = self.datapropare()
        Wghlist2 = self.weight()
        errvalue = 1
        while abs(errvalue) >= self.error:
            for i in dataSet:
                accumValue = self.accumulation(i, Wghlist2)
                PredictValue = self.Activationfun(accumValue)
                errvalue = self.deffiValue(PredictValue, target[dataSet.index(i)])
                #errvalue += errvalue1
            #errvalue = errvalue/(len(dataSet))
                for ii in range(3):
                    Wghlist2[ii] = self.adjustWght(errvalue,Wghlist2[ii],i[ii])
                print(errvalue)
        return Wghlist2

    def Forecasting(self,data):
        Wghlist2 = self.perceptron()
        sum = 0
        for i in data:
            sum += i*Wghlist2[data.index(i)]
        sum +=Wghlist2[2]
        if sum > 0:
            targets = 1
        else:
            targets = 0
        return targets

if __name__=="__main__":
    newNN = perceptron()
    newNN.perceptron()
    while True:
        data = []
        data1=input("请输入第一个逻辑（1 = True, 2=false)：")
        data2 = input("请输入第一个逻辑（1 = True, 2=false)：")
        data.append(int(data1))
        data.append(int(data2))
        targets = newNN.Forecasting(data)
        print("结果是(1 = True, 2=false):"+ str(targets) )
