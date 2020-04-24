# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance

### load datasets
digits = datasets.load_digits()

### data analysis
print(digits.data.shape)
print(digits.target.shape)

### data split
x_train, x_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.3,
                                                    random_state=33)

model = XGBClassifier()
print('type(x_train),type(y_train)',type(x_train),x_train.shape,type(y_train),y_train.shape)
model.fit(x_train, y_train)
print('model',model,type(model))
### plot feature importance
fig, ax = plt.subplots(figsize=(15, 15))
plot_importance(model,
                height=0.5,
                ax=ax,
                max_num_features=66)
plt.show()

### make prediction for test data
y_pred = model.predict(x_test)

### model evaluate
accuracy = accuracy_score(y_test, y_pred)
print("accuarcy: %.2f%%" % (accuracy * 100.0))