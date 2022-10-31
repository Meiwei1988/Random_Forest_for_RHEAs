# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:36:58 2021

@author: Maywell2019
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

def compareResults(a,b):
    if a > b :
        return b
    else:
        return a

dataset = pd.read_csv('Total_results.csv')

x = dataset.iloc[:,2:9].values
y = dataset['AVR']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size =0.1, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
R2 = metrics.r2_score(y_test,y_pred)

print('Mean Absolute Error:', MAE)
print('Mean Squared Error:', MSE)
print('Root Mean Squared Error:', RMSE)
print('R2:',R2)

x_min = np.min(y_test)
x_max = np.max(y_test)
y_min = np.min(y_pred)
y_max = np.max(y_pred)
axix_range = [compareResults(x_min, y_min),compareResults(x_max, y_max)]
axix_length = axix_range[1] - axix_range[0]

font1 = {'family' : 'Times New Roman',
'style' : 'normal',
'size'   : 18,
}

plt.figure(figsize=(5, 5))
plt.scatter(y_test,y_pred,marker='o',c='',edgecolors='blue')
plt.xlim(axix_range)
plt.ylim(axix_range)
### Drawing the guide line
plt.plot(axix_range, axix_range,c='r',linewidth=3, linestyle="--" )

plt.xlabel(r'Calculation AVR(GPa)',font1)
plt.ylabel('Prediction AVR(GPa)',font1)
plt.xticks(fontproperties = 'Times New Roman', size = 12)
plt.yticks(fontproperties = 'Times New Roman', size = 12)
plt.text(axix_range[0]+axix_length*0.05,axix_range[1]-axix_length*0.1, '$\mathregular{R^2}$=' + '%0.6f'% R2,fontproperties = 'Times New Roman', size = 15,)
plt.savefig('./AVR.png',bbox_inches = 'tight',dpi=600)
plt.show()
