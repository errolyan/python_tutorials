# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  你命由你，不由天。若你觉得命运不公，那就和它抗争到底。
@Evn     :  
@Date    :  2019-08-23  16:06
'''

import numpy as np
import pandas as pd

fortune = np.ones(100) * 100
people_id = range(100)

p = np.ones(100) * 1/(1.02*10+1*90)
for i in range(10):
    p[i] = 1/(1.02*10+1*90) * 1.02

days = 35*365
for i in range(days):
    fortune = fortune - np.ones(100)
    receivers = np.random.choice(people_id, 100, p=p)
    unique, counts = np.unique(receivers, return_counts=True)
    gains = zip(unique, counts)
    for (idx, gain) in gains:
        fortune[idx] += gain

print(fortune)

