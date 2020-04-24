# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  pip install cufflinks  https://plot.ly/ipython-notebooks/cufflinks/
@Evn     :  
@Date    :  2019-07-25  21:31
'''
import plotly.offline as py_offline
import cufflinks as cf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objs as go


# set figure size
mpl.rcParams['figure.figsize']=[15,12]

# set figure style
plt.style.use('ggplot')

# plotly layout
layout=go.Layout(
    autosize=True,
    margin=go.layout.Margin(
        l=20,
        r=20,
        t=40,
        b=20,
        pad=10),
    template='ggplot2'
)

py_offline.init_notebook_mode(connected=True)
print(cf.__version__)
numbers = np.random.standard_normal((259,5)).cumsum(axis=0)

# convert to time series data
index = pd.date_range('2019-01-01', freq='B', periods=len(numbers))

df = pd.DataFrame(100+5*numbers, columns=list('abcde'), index=index)

df.plot()
plt.show()



trace = []
for column in df.columns:
    temp = go.Scatter(
        x=df.index,
        y=df[column],
        mode='lines',
        name=column
    )
    trace.append(temp)

layout=go.Layout(title='A Time Series Plot',
                                  xaxis={'title':'Date'},
                                  yaxis={'title':'Value'},
                                  template='ggplot2')
py_offline.iplot(go.Figure(trace, layout))

from IPython.display import display, HTML
import pandas as pd
import numpy as np
import math

# Using plotly + cufflinks in offline mode

import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True)
df = pd.read_csv('data/restaurants.txt')
df = df.groupby(by="Country").count()["Name"]
df = df.sort_values(ascending=False)
df.plot(kind='bar')

plt.show()
