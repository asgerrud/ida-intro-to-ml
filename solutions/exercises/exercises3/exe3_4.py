import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df = pd.read_csv('../sa_heart.csv')
famhist = pd.get_dummies(df['famhist'])
df = df.join(famhist)
df = df.drop('famhist', axis=1)

X = df.drop('chd', axis=1)
X = X.to_numpy()
y = df['chd']
y = y.to_numpy()

kf = KFold(n_splits=10, shuffle=True)

perf_list = []
for d in range(1,6+1):
    fold_list = []
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        model = DecisionTreeClassifier(max_depth=d).fit(train_x, train_y)
        pred_y = model.predict(test_x)
        acc = accuracy_score(test_y,  pred_y)
        fold_list.append(acc)
    perf_list.append(np.mean(fold_list))

print(list(range(1,6+1)))
print(perf_list)