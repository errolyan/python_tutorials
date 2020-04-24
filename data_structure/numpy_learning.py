#! /bin/python
# coding = "utf-8"
# Author == "ErrolYan"

from numpy import *

# 产生随机数组
A = random.rand(4,4)
print(A,type(A))

# 数组转为矩阵
randmatA = mat(A)
print(randmatA,type(randmatA))

# 矩阵求逆
invrandmat = randmatA.I
print(invrandmat,type(invrandmat))

#矩阵乘法(矩阵与逆矩阵相乘）
matB = randmatA*invrandmat
print(matB,type(matB))

matC = matB - eye(4)
print(matC)