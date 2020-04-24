# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import urllib
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/practicalAI/practicalAI/master/data/titanic.csv'
response = urllib.request.urlopen(url)
html = response.read()
with open('titanic.csv', 'wb') as f:
    f.write(html)

# Read from CSV to Pandas DataFrame
df = pd.read_csv('titanic.csv', header=0)

# First five items
print(df.head())
print(df.describe())

# 协方差
plt.matshow(df.corr())
continuous_features = df.describe().columns
plt.xticks(range(len(continuous_features)), continuous_features, rotation='45')
plt.yticks(range(len(continuous_features)), continuous_features, rotation='45')
plt.colorbar()
plt.show()

# Histograms
df['age'].hist()


# Unique values
df['embarked'].unique()

# Filtering
df[df['sex']=='female'].head()

# Grouping
survived_group = df.groupby('survived')
print(survived_group.mean())

# Selecting row 0
df.iloc[0, :]

# Selecting a specific value
df.iloc[0, 1]

# Drop rows with Nan values
df = df.dropna() # removes rows with any NaN values
df = df.reset_index() # reset's row indexes in case any rows were dropped
print(df.head())

# Map feature values
df['sex'] = df['sex'].map( {'female': 0, 'male': 1} ).astype(int)
df['embarked'] = df['embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)
print(df.head())