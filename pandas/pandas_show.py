# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  
@Email:2681506@gmail.com   
@Date:  2019-05-27  11:18
@File：pandas_show.py
@Describe:数据可视化
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition

sns.set(style="white", color_codes=True)

#加载iris数据集
from sklearn.datasets import load_iris
iris_data = load_iris()
iris = pd.DataFrame(iris_data['data'], columns=iris_data['feature_names'])
print(iris)
iris = pd.merge(iris, pd.DataFrame(iris_data['target'], columns=['species']), left_index=True, right_index=True)
print(iris)
labels = dict(zip([0,1,2], iris_data['target_names']))
iris['species'] = iris['species'].apply(lambda x: labels[x])
iris.head()


'''
用boxplot画出单个特征与因变量的关系
'''
sns.boxplot(x='species', y='petal length (cm)', data=iris)
plt.title("boxplot img")
plt.savefig('boxplot.png')
plt.show()


# kdeplot核密度图
# kdeplot looking at univariate relations
# creates and visualizes a kernel density estimate of the underlying feature

sns.FacetGrid(iris, hue='species',size=6) \
   .map(sns.kdeplot, 'petal length (cm)') \
    .add_legend()
plt.title("kdeplot img")
plt.show()

# violinplot琴形图：结合了箱线图与核密度估计图的特点，它表征了在一个或多个分类变量情况下，连续变量数据的分布并进行了比较，它是一种观察多个数据分布有效方法。
# A violin plot combines the benefits of the boxplot and kdeplot
# Denser regions of the data are fatter, and sparser thiner in a violin plot

sns.violinplot(x='species', y='petal length (cm)', data=iris, size=6)
plt.title("violinplot img")
plt.show()


'''
二维数组
'''
# use seaborn's FacetGrid to color the scatterplot by species

sns.FacetGrid(iris, hue="species", size=5) \
    .map(plt.scatter, "sepal length (cm)", "sepal width (cm)") \
    .add_legend()
plt.title("FacetGrid  img")
plt.show()

'''
pairplot：展现特征的两两关系
'''
# pairplot shows the bivariate relation between each pair of features
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other two across all feature combinations
# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde

sns.pairplot(iris, hue='species', size=3, diag_kind='kde')
plt.title("pairplot")
plt.show()


'''
多维数据可视化
'''
# Andrews曲线将每个样本的属性值转化为傅里叶序列的系数来创建曲线。通过将每一类曲线标成不同颜色可以可视化聚类数据，属于相同类别的样本的曲线通常更加接近并构成了更大的结构。
# Andrews Curves involve using attributes of samples as coefficients for Fourier series and then plotting these

pd.plotting.andrews_curves(iris, 'species')
plt.title("Andrews")
plt.show()

'''
 平行坐标

平行坐标也是一种多维可视化技术。它可以看到数据中的类别以及从视觉上估计其他的统计量。使用平行坐标时，每个点用线段联接。每个垂直的线代表一个属性。一组联接的线段表示一个数据点。可能是一类的数据点会更加接近。
'''

# Parallel coordinates plots each feature on a separate column & then draws lines connecting the features for each data sample

pd.plotting.parallel_coordinates(iris, 'species')
plt.title("Parallel")
plt.show()


'''
RadViz是一种可视化多维数据的方式。它基于基本的弹簧压力最小化算法（在复杂网络分析中也会经常应用）。
简单来说，将一组点放在一个平面上，每一个点代表一个属性，我们案例中有四个点，被放在一个单位圆上，接下来你可以设想每个数据集通过一个弹簧联接到每个点上，
弹力和他们属性值成正比（属性值已经标准化），数据集在平面上的位置是弹簧的均衡位置。不同类的样本用不同颜色表示。
'''
# radviz  puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted by the relative value for that feature

pd.plotting.radviz(iris, 'species')
plt.title("radviz")
plt.show()

'''
因子分析FactorAnalysis是指研究从变量群中提取共性因子的统计技术。
最早由英国心理学家C.E.斯皮尔曼提出。他发现学生的各科成绩之间存在着一定的相关性，一科成绩好的学生，
往往其他各科成绩也比较好，从而推想是否存在某些潜在的共性因子，或称某些一般智力条件影响着学生的学习成绩。
因子分析可在许多变量中找出隐藏的具有代表性的因子。将相同本质的变量归入一个因子，可减少变量的数目，还可检验变量间关系的假设。

'''
fa = decomposition.FactorAnalysis(n_components=2)
X = fa.fit_transform(iris.iloc[:,:-1].values)

pos=pd.DataFrame()
pos['X'] =X[:, 0]
pos['Y'] =X[:, 1]
pos['species'] = iris['species']

ax = pos[pos['species']=='virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='virginica')
pos[pos['species']=='setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='setosa', ax=ax)
pos[pos['species']=='versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='versicolor', ax=ax)

plt.title("FactorAnalysis")
plt.show()


'''
主成分分析是由因子分析进化而来的一种降维的方法，通过正交变换将原始特征转换为线性独立的特征，转换后得到的特征被称为主成分。
主成分分析可以将原始维度降维到n个维度，有一个特例情况，就是通过主成分分析将维度降低为2维，这样的话，就可以将多维数据转换为平面中的点，
来达到多维数据可视化的目的。
'''
pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(iris.iloc[:,:-1].values)

pos=pd.DataFrame()
pos['X'] =X[:, 0]
pos['Y'] =X[:, 1]
pos['species'] = iris['species']

ax = pos[pos['species']=='virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='virginica')
pos[pos['species']=='setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='setosa', ax=ax)
pos[pos['species']=='versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='versicolor', ax=ax)
plt.title("PCA")
plt.show()
print("保留的两个主成分可以解释原始数据的多少：",pca.fit(iris.iloc[:,:-1].values).explained_variance_ratio_)

'''
独立成分分析将多源信号拆分成最大可能独立性的子成分，它最初不是用来降维，而是用于拆分重叠的信号。
'''
fica = decomposition.FastICA(n_components=2)
X = fica.fit_transform(iris.iloc[:,:-1].values)
pos=pd.DataFrame()
pos['X'] =X[:, 0]
pos['Y'] =X[:, 1]
pos['species'] = iris['species']

ax = pos[pos['species']=='virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='virginica')
pos[pos['species']=='setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='setosa', ax=ax)
pos[pos['species']=='versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='versicolor', ax=ax)
plt.title("ICA")
plt.show()


'''
多维度量尺被用于数据的相似性，它试图用几何空间中的距离来建模数据的相似性，直白来说就是用二维空间中的距离来表示高维空间的关系。
数据可以是物体之间的相似度、分子之间的交互频率或国家间交易指数。这一点与前面的方法不同，前面的方法的输入都是原始数据，而在多维度量尺的例子中，
输入是基于欧式距离的距离矩阵。多维度量尺算法是一个不断迭代的过程，因此，需要使用max_iter来指定最大迭代次数，同时计算的耗时也是上面算法中最大的一个。 
'''
from sklearn import manifold

from sklearn.metrics import euclidean_distances

similarities = euclidean_distances(iris.iloc[:,:-1].values)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
X = mds.fit(similarities).embedding_

pos=pd.DataFrame(X, columns=['X', 'Y'])
pos['species'] = iris['species']

ax = pos[pos['species']=='virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='virginica')
pos[pos['species']=='setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='setosa', ax=ax)
pos[pos['species']=='versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='versicolor', ax=ax)
plt.title("Multi-dimensional scaling, MDS")
plt.show()

'''
t-SNE(t分布随机邻域嵌入)是一种用于探索高维数据的非线性降维算法。
通过基于具有多个特征的数据点的相似性识别观察到的簇来在数据中找到模式，将多维数据映射到适合于人类观察的两个或多个维度。
本质上是一种降维和可视化技术。使用该算法的最佳方法是将其用于探索性数据分析
'''
from sklearn.manifold import TSNE

iris_embedded = TSNE(n_components=2).fit_transform(iris.iloc[:,:-1])

pos = pd.DataFrame(iris_embedded, columns=['X','Y'])
pos['species'] = iris['species']

ax = pos[pos['species']=='virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='virgnica')
pos[pos['species']=='setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='setosa', ax=ax)
pos[pos['species']=='versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='versicolor', ax=ax)
plt.title("t-distributed Stochastic Neighbor Embedding")
plt.show()
