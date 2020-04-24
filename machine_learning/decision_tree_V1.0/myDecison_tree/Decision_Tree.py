#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "errrolyan"
# __Date__: 18-12-10
# __Describe__ = "决策树ID3算法算法Python实现版本”


from math import log
import operator
import copy


class DecisionTree():

    def __init__(self, datapath):
        self.datapath = datapath

    def dataprocess(self):
        with open(self.datapath, "r") as Rdata:
            rdata = Rdata.read()
            DTSetAll = []
            list = rdata.split("\n")
            for element in list:
                if element != "":
                    DTSetAll.append(element.split(","))
        AttributList = DTSetAll[0][1:-1]
        DTSetlist = DTSetAll[1:]
        return AttributList,DTSetlist,DTSetAll


    def calcShannonEnt(self,DTSetlist,DTSetAll):
        SampleNum = len(DTSetlist)
        # 统计类别数量
        AttriClassList = []
        for i in range(SampleNum):
            AttriClassList.append(DTSetlist[i][-1])
            i += 1
        AttriClassSet = set(AttriClassList)
        # 类别和类的数量存储在字典classesDic中,classesDic的key 是类别，classesDic为value为类的数量。
        classesDic = {}
        for Attri in AttriClassSet:
            classesDic[Attri] = 0
        for Attri in classesDic.keys():
            for j in AttriClassList:
                if Attri == j:
                    classesDic[j] += 1
        # 计算跟结点总的熵
        shannonEnt = 0
        for i in classesDic.keys():
            shannonEnt += -(classesDic[i]/SampleNum)* log(classesDic[i]/SampleNum, 2)
        return shannonEnt

    def classEntor(self, Attribut, dataset,AttributList,DTSetAll):
        key_num = AttributList.index(Attribut) + 1
        classlis = []

        for i in range(len(dataset)):
            classlis.append(dataset[i][key_num])

        ClassSet = set(classlis)
        classDic = {}
        for cls in ClassSet:
            classDic[cls] = 0
            for clsi in classlis:
                if cls == clsi:
                    classDic[cls] += 1
        classEnt = 0
        for i in classDic.keys():
            prot = classDic[i]/len(dataset)
            classEnt += -prot * log(prot, 2)
        return classEnt



    def Choose_Feature(self, attributlist, dataset,AttributList,DTSetAll):
        shannonEnt = self.calcShannonEnt(DTSetAll[1:],DTSetAll)
        print(shannonEnt)
        information_gain_Dic = {}
        for keyfeat in attributlist:
            Differ_value = shannonEnt - self.classEntor(keyfeat, dataset,AttributList,DTSetAll)
            information_gain_Dic[keyfeat] = Differ_value
        bestFeat = max(information_gain_Dic)
        print(information_gain_Dic)
        return bestFeat

    def Classification(self, bestfeat, dataset,AttributList,DTSetAll):
        datasetdic = {}
        classlis = []
        key_num = AttributList.index(bestfeat) + 1
        for i in range(len(dataset)):
            classlis.append(dataset[i][key_num])

        ClassSet = set(classlis)
        for i in ClassSet:
            list = []
            for j in range(len(dataset)):
                if i in dataset[j]:
                    list.append(dataset[j])
            datasetdic[i] = list
        return ClassSet, datasetdic

    def Decision_Tree(self):
        AttributList, DTSetlist, DTSetAll = self.dataprocess()
        attributlist = copy.deepcopy(AttributList)
        #print(attributlist)
        bestfeat = self.Choose_Feature(attributlist, DTSetlist, AttributList,DTSetAll)
        DTree = {bestfeat: {}}
        attributlist.remove(bestfeat)
        #print(attributlist)
        ClassSet, DatasetDic = self.Classification(bestfeat,DTSetlist, AttributList, DTSetAll)
        #print(ClassSet, DatasetDic)
        for featkeys in DatasetDic.keys():
            if len(DatasetDic[featkeys]) != 1:
                DTree[bestfeat][featkeys] = {}
            else:
                DTree[bestfeat][featkeys] = DatasetDic[featkeys][-1]
        #print(DTree)
        print(DatasetDic)
        list = DatasetDic.keys()
        for featkeys1 in list:
            dataset = DatasetDic[featkeys1]
            #print(dataset)
            bestfeat = self.Choose_Feature(attributlist, dataset, AttributList, DTSetAll)
            #print(bestfeat)
            ClassSet, DatasetDic = self.Classification(bestfeat, dataset, AttributList, DTSetAll)
            print(ClassSet)
            print(DatasetDic)


if __name__ == "__main__":
    # 年龄分为 30>岁 A，30~40岁 B，40岁< C
    # 收入分为 20万> 低收入 ，20万~30万 中等收入 ，30万< 高收入
    newdata = DecisionTree("./DataSet/dataset.csv")
    newdata.Decision_Tree()

