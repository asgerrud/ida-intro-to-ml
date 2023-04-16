import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
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

# train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.20)
# model = DecisionTreeClassifier().fit(train_x, train_y)
# pred_y = model.predict(test_x)
# accuracy = accuracy_score(test_y, pred_y)
# print("Baseline: ", baseline)
# print(accuracy)

kf = KFold(n_splits=10, shuffle=True)

acc_list = []
for train_index, test_index in kf.split(X):
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    model = DecisionTreeClassifier().fit(train_x, train_y)
    pred_y = model.predict(test_x)
    rmse = accuracy_score(test_y,  pred_y)
    acc_list.append(rmse)

print("Baseline: ", baseline)
print(np.mean(acc_list))