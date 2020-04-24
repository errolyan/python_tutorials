# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  pip install seaborn
@Evn     :  
@Date    :  2019-07-28  11:29
'''

'''
数据可视化
'''
import matplotlib.pyplot as plt
from numpy.random import normal
x = normal(size=100)
plt.hist(x, bins=20)
plt.savefig("./1.png")
#plt.show(1)


from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-10, 10, 0.1)
Y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
plt.savefig("./2.png")
#plt.show(1)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
print(tips,type(tips))
print(tips.head(3))
g=sns.relplot(x="total_bill", y="tip", col="time",hue="smoker", style="smoker", size="size",data=tips);
g.savefig("./3.png")
f = sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips)
f.savefig("./4.png")


df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.savefig('./5.png')

df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
g = sns.relplot(x="x", y="y", sort=False, kind="line", data=df)
g.savefig('./6.png')


fmri = sns.load_dataset("fmri")
g= sns.relplot(x="timepoint", y="signal", hue="region",
            units="subject", estimator=None,
            kind="line", data=fmri.query("event == 'stim'"))
g.savefig('./7.png')


'''
分类数据
'''

tips = sns.load_dataset("tips")
fig = sns.catplot(x="day", y="total_bill", data=tips)
fig.savefig("./8.png")

fig = sns.catplot(x="day", y="total_bill", kind="box", data=tips)
fig.savefig('./9.png')

fig = sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips)
fig.savefig('./10.png')

'''
Violinplots
'''
fig = sns.catplot(x="total_bill", y="day", hue="time",
            kind="violin", data=tips)
fig.savefig('./11.png')

fig = sns.catplot(x="day", y="total_bill", hue="sex",
            kind="violin", inner="stick", split=True,
            palette="pastel", data=tips)
fig.savefig('./12.png')

'''
Bar plots
'''
titanic = sns.load_dataset("titanic")
fig = sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)
fig.savefig('./13')

fig = sns.catplot(x="day", y="total_bill", hue="smoker",
            col="time", aspect=.6,
            kind="swarm", data=tips)
fig.savefig('./14.png')



'''
Visualizing pairwise relationships
'''
iris = sns.load_dataset("iris")

print(iris,type(iris))
print(iris.head(3))
fig = sns.pairplot(iris)
print(fig,type(fig))
fig.savefig('./15.png')

g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6)
g.savefig('./16.png')

'''
Visualizing linear relationships
'''
sns.set(color_codes=True)
tips = sns.load_dataset("tips")
fig = sns.regplot(x="total_bill", y="tip", data=tips)
print(fig,type(fig))

anscombe = sns.load_dataset("anscombe")
fig = sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           robust=True, ci=None, scatter_kws={"s": 80})
fig.savefig('./17.png')

fig = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1")
fig.savefig('./18.png')


fig = sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", height=5, aspect=.8, kind="reg")

fig.savefig('./19.png')

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time")
g = sns.FacetGrid(tips, col="time")
g.map(plt.hist, "tip")
g.savefig('./20.png')

g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(plt.scatter, "total_bill", "tip", alpha=.7)
g.add_legend()
g.savefig('./21.png')


g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
g.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=False, x_jitter=.1)
g.savefig('./22.png')

g = sns.FacetGrid(tips, hue="sex", palette="Set1", height=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "total_bill", "tip", s=100, linewidth=.5, edgecolor="white")
g.add_legend()
g.savefig('./23.png')

with sns.axes_style("white"):
    g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, height=2.5)
g.map(plt.scatter, "total_bill", "tip", color="#334488", edgecolor="white", lw=.5);
g.set_axis_labels("Total bill (US Dollars)", "Tip");
g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);
g.fig.subplots_adjust(wspace=.02, hspace=.02);
g.savefig('./24.png')


g = sns.PairGrid(iris, hue="species")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();
g.savefig('./25.png')

g = sns.PairGrid(iris)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)
g.savefig('./26.png')


g = sns.PairGrid(tips, hue="size", palette="GnBu_d")
g.map(plt.scatter, s=50, edgecolor="white")
g.add_legend();
g.savefig('./27.png')

fig = sns.pairplot(iris, hue="species", height=2.5);
fig.savefig('./28.png')

g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", height=2.5)
g.savefig('./29.png')