import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df = pd.read_csv('../winequality-red.csv', sep=";")
df['goodquality'] = [1 if x >= 6 else 0 for x in df['quality']]
df = df.drop('quality', axis=1)

X = df.drop('goodquality', axis=1)
X = X.to_numpy()
y = df['goodquality']
y = y.to_numpy()

gq_counts = df['goodquality'].value_counts() 
baseline = max(gq_counts) / (gq_counts[0] + gq_counts[1])

kf = KFold(n_splits=10, shuffle=True)

acc_list_dt = []
acc_list_log = []
for train_index, test_index in kf.split(X):
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    
    model_dt = DecisionTreeClassifier().fit(train_x, train_y)
    pred_y = model_dt.predict(test_x)
    acc = accuracy_score(test_y,  pred_y)
    acc_list_dt.append(acc)

    model_log = LogisticRegression(max_iter=1000).fit(train_x, train_y)
    pred_y = model_log.predict(test_x)
    acc = accuracy_score(test_y,  pred_y)
    acc_list_log.append(acc)

print("DT: ",np.mean(acc_list_dt))
print("log: ",np.mean(acc_list_log))