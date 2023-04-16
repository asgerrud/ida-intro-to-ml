
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# We need to predict if a person will get a heart disease or not

df = pd.read_csv("data/sa_heart.csv", sep=",")

# One-hot encode famhist column
dummies = pd.get_dummies(df['famhist'])
df = df.join(dummies)
df = df.drop('famhist', axis=1)

# X og Y
X = df[["sbp", "tobacco", "ldl", "adiposity",
        "Present", "Absent", "typea", "obesity", "alcohol"]]
X = X.to_numpy()
y = df['chd']
y = y.to_numpy()

# Get baseline
rows_per_class = df['chd'].value_counts()  # 0=302; 1=160
baseline = 302/len(df)  # 0.65%, assuming no (chd=0) to have a heart disease

# 80/20 split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

model = LogisticRegression().fit(train_x, train_y)

pred_y = model.predict(test_x)

accuracy = accuracy_score(test_y, pred_y)

print(accuracy)  # 0.69 - 0.78. Good enough
