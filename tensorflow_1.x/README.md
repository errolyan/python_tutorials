[TOC]
# tensorflow 2.0 中文教程
## 环境搭建
1.anaconda
```angular2html
conda create --name python3tutorial python=3.6

#激活当前环境
source activate python3tutorial 

#关闭当前环境
source deactivate 

#删除当前环境
conda remove -n python3tutorial --all

```
2.tensorflow 2.0
```angular2html
pip install tensorflow==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple

```

3.pytorch 1.0 
```angular2html
pip install torch==1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

```
## docker
1.docker images
```angular2html
cd ./docker
sudo docker build -t rep:version .
```