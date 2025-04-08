import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./articulos_ml.csv")
#print(data.shape)
#print(data.head())

#print(data.describe())

#data.drop(columns=['Title', 'url', 'Elapsed days']).hist()
#plt.show()

filtered_data = data[(data['Word count']<=3500)&(data['# Shares'] <= 80000)]
colores = ['orange', 'blue']
tamanios = [30, 60]
f1 = filtered_data['Word count'].values
f2 = filtered_data['# Shares'].values

asignar=[] 
for index, row in filtered_data.iterrows():
    if(row['Word count'] > 1808):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
plt.scatter(f1, f2, c = asignar, s = tamanios[0]) 
plt.show()

dataX = filtered_data[["Word count"]]
X_train = np.array(dataX)
y_train = filtered_data['# Shares'].values
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_train)

print('Coefficients:',regr.coef_)
print('Independent term:',regr.intercept_)
print("Mean squared error:%.2f"%mean_squared_error(y_train,y_pred))
print('Variance score:%.2f'%r2_score(y_train,y_pred))

y_Dosmil = regr.predict([[2000]])
print(int(y_Dosmil))