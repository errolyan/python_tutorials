# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV

def knn_class():
    '''
    KNN 分类
    :return:
    '''
    # 读取数据
    data = pd.read_csv("./train.csv")
    print(data)

    # 数据拆分

    y = data['xingbie']
    x = data.drop("xingbie",axis=1)

    # 特征工程
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.1)
    print(x_train,x_test,y_train,y_test)
    ## 标准化：对训练集和测试集的特征值进行归一化
    minmax = MinMaxScaler()
    x_train = minmax.fit_transform(x_train)
    x_test = minmax.transform(x_test)

    # 模型训练
    knn = KNeighborsClassifier(n_neighbors=2)
    # knn.fit(x_train,y_train)
    # y_predic = knn.predict(x_test)
    # print('y_precdic',y_predic)
    #
    #
    # # 准确率
    #
    # print("预测准确率",knn.score(x_test,y_test))

    # 训练集上交叉验证
    gc = GridSearchCV(knn,param_grid = {
        "n_neighbors":[1,2,3]
    },cv = 2)
    gc.fit(x_train,y_train)

    # 测试集上测试分数
    score = gc.score(x_test,y_test)
    print('测试集上准确率：',score)
    print("交叉验证中最好的结果：",gc.best_params_)
    print("选择最好的模型：",gc.best_params_)
    print("每次交叉验证的结果", gc.cv_results_)
    return None

if __name__ == "__main__":
    knn_class()