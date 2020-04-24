# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''

# 特征抽取--字典格式转为向量
from sklearn.feature_extraction import DictVectorizer
def dict2Vect():
    onehot = DictVectorizer(sparse=True) # 如果结果不用toarray，请开启sparse=False
    instances = [{'city': '北京','temperature':100},{'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    data = onehot.fit_transform(instances)
    print('data',data)
    X = data.toarray()
    print("X", X)
    print(onehot.inverse_transform(X))

dict2Vect()

# 文本特征提取（文本转化为向量
from sklearn.feature_extraction.text import CountVectorizer

def text2vect():
    content = ["life is is short,i like python", "life is too long,i dislike python"]
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(content).toarray()
    print(vectorizer.get_feature_names())
    print('data',data)

text2vect()

from sklearn.feature_extraction.text import CountVectorizer
import jieba
def hanzi2vect():
    content1 = list(jieba.cut("我们爱中国，我们不喜欢美国"))
    content2 = list(jieba.cut("我们是编程人员，再是算法研究员"))
    content1 = ' '.join(content1)
    content2 = ' '.join(content2)
    content = []
    content.append(content1)
    content.append(content2)

    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(content).toarray()
    print(vectorizer.get_feature_names())
    print('data', data)

hanzi2vect()


# 文本分类
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
def tf_idf():
    content1 = list(jieba.cut("我们爱中国，我们不喜欢美国"))
    content2 = list(jieba.cut("我们是编程人员，再是算法研究员"))
    content1 = ' '.join(content1)
    content2 = ' '.join(content2)
    content = []
    content.append(content1)
    content.append(content2)

    tf_idfver = TfidfVectorizer()
    data = tf_idfver.fit_transform(content).toarray()
    print(tf_idfver.get_feature_names())
    print('data', data)

tf_idf()

# minmaxstander
from sklearn.preprocessing import MinMaxScaler
def minmax():
    '''
    归一化
    :return:
    '''
    mm = MinMaxScaler(feature_range=(2,4))
    data = mm.fit_transform([[90,2,15],[60,1,20],[70,1.5,18]])
    print('minmax',data)
minmax()

# 标准化缩放
from sklearn.preprocessing import StandardScaler
def stander_sacler():
    '''
    标准化缩放
    :return:
    '''
    std = StandardScaler()
    data = std.fit_transform([[1,2,3],[2,3,4],[4,5,6]])
    print(data)
    return 0

stander_sacler()

# 特征选择
from sklearn.feature_selection import VarianceThreshold
def function_select():
    '''
    删除低方差的特征
    :return:
    '''
    fun_select = VarianceThreshold()
    data = fun_select.fit_transform([[1,2,3],[1,3,4],[1,4,5]])
    print(data)

function_select()

# 主成分分析
from sklearn.decomposition import PCA
def pca_fun():
    '''
    主成分分析
    :return:
    '''
    pac =PCA(n_components=1)# 信息保存的比例
    data = pac.fit_transform([[1,2,3],[2,3,4]])
    print('pca',data)


pca_fun()


#缺失值处理

from sklearn.preprocessing import Imputer
import numpy as np
def imputer():
    '''
    缺失值处理
    :return:
    '''
    im = Imputer(missing_values='NaN',strategy = 'mean',axis =0)
    data = im.fit_transform([[1,2],[np.nan,3],[7,6]])
    print('缺失值处理',data)
    return None

imputer()

from sklearn.feature_selection import VarianceThreshold
# 特征选择-删除低方差的特征
def var():
    '''
    特征选择的依据
    :return:
    '''
    var = VarianceThreshold(threshold=1.0)

    data = var.fit_transform([[0,2,3,4],[1,0,2,3],[5,6,7,0]])

    print('低var删除',data)
    return None

var()

