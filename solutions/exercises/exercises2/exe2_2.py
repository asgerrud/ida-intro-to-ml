import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

df = pd.read_csv('../sa_heart.csv', sep=",")
famhist = pd.get_dummies(df['famhist'])
df = df.join(famhist)
df = df.drop('famhist', axis=1)
X = df.drop('obesity',axis=1)
X = X.to_numpy()
y = df['obesity']
y = y.to_numpy()

# train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.20)

# model = LinearRegression().fit(train_x, train_y)

# pred_y = model.predict(test_x)

# print(mean_squared_error(test_y, pred_y))

kf = KFold(n_splits=10, shuffle=True)

rmse_list = []
for train_index, test_index in kf.split(X):
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    model = LinearRegression().fit(train_x, train_y)
    pred_y = model.predict(test_x)
    rmse = mean_squared_error(test_y,  pred_y, squared=False)
    rmse_list.append(rmse)

print(np.mean(rmse_list))