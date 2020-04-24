# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： seaborn_learning3
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/1  23:40
-------------------------------------------------
   Change Activity:
                  2020/2/1  23:40:
-------------------------------------------------
'''
__author__ = 'yanerrol'
import matplotlib.pyplot as plt
import scipy.stats as stats

#model2 is a regression model
log_resid = model2.predict(X_test)-y_test
stats.probplot(log_resid, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()