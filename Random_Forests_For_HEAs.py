#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:44:29 2021

@author: wei
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def compareResults(a,b):
    if a > b :
        return b
    else:
        return a

dataset1 = pd.read_csv('non_equiatomic_HEAs.csv')
dataset1.head()

dataset2 = pd.read_csv('five_elements_HEAs.csv')
dataset2.head()

dataset3 = pd.read_csv('four_elements_HEAs.csv')
dataset3.head()

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

x = np.concatenate((dataset1.iloc[:,1:6].values,dataset2.iloc[:,1:6].values,dataset3.iloc[:,1:6].values))
y = np.concatenate((dataset1.iloc[:,7].values,dataset2.iloc[:,7].values,dataset3.iloc[:,7].values))

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size =0.1, random_state=0)

#x_test = dataset2.iloc[:,1:6].values
#y_test = dataset2.iloc[:,6].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', MAE)
print('Mean Squared Error:', MSE)
print('Root Mean Squared Error:', RMSE)

x_min = np.min(y_test)
x_max = np.max(y_test)
y_min = np.min(y_pred)
y_max = np.max(y_pred)
axix_range = [compareResults(x_min, y_min),compareResults(x_max, y_max)]

import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.scatter(y_test,y_pred,marker='o',c='',edgecolors='blue')
plt.xlim(axix_range)
plt.ylim(axix_range)
### Drawing the guide line
plt.plot(axix_range, axix_range,c='r',linewidth=3, linestyle="--" )

plt.xlabel('Calculation B(GPa)',font1)
plt.ylabel('Prediction B(GPa)',font1)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
#plt.text(140,222, 'RMSE=' + str(RMSE),fontproperties = 'Times New Roman', size = 15,)
#plt.savefig('./B_RF_new.png',bbox_inches = 'tight',dpi=600)
plt.show()
