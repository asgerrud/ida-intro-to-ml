import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df = pd.read_csv('../sa_heart.csv', sep=",")
famhist = pd.get_dummies(df['famhist'])
df = df.join(famhist)
df = df.drop('famhist', axis=1)
############################
# Drop all these to get better performance
df = df.drop('tobacco',axis=1)
df = df.drop('alcohol', axis=1)
df = df.drop('ldl', axis=1)
df = df.drop('sbp', axis=1)
df = df.drop('chd', axis=1)
df = df.drop('Absent', axis=1)
df = df.drop('Present', axis=1)
############################
X_df = df.drop('obesity',axis=1)
X = X_df.to_numpy()
y_df = df['obesity']
y = y_df.to_numpy()

print(X_df.columns)

kf = KFold(n_splits=10, shuffle=True, random_state=3)

rmse_list = []
for train_index, test_index in kf.split(X):
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    model = LinearRegression().fit(train_x, train_y)
    pred_y = model.predict(test_x)
    rmse = mean_squared_error(test_y,  pred_y, squared=False)
    rmse_list.append(rmse)

print(np.mean(rmse_list))

#drop famhist = 2.719 
#drop tobacco = 2.713
#drop alcohol = 2.710
#drop ldl = 2.707
#drop sbp = 2.7008
#drop chd = 2.699
#drop Absent+Present  = 2.695