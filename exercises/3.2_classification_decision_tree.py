
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns

# We want to predict if a wine is good or not

df = pd.read_csv("data/winequality-red.csv", sep=";")

# Add new column based on other column
df['good_quality'] = [1 if x >= 6 else 0 for x in df['quality']]
df = df.drop('quality', axis=1)

X = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
X = X.to_numpy()
y = df['good_quality']
y = y.to_numpy()

# Get baseline
rows_per_class = df['good_quality'].value_counts()  # 0=744; 1=855
baseline = 855/len(df)  # 0.535, Assuming most wines are good quality

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42)

accuracies = []  # We expect around 73% when depth is not set (so default?)


# 3.3 find depth which yields highest accuracy
max_depth = 20
for i in range(1, max_depth + 1):
    model = DecisionTreeClassifier(max_depth=i).fit(train_x, train_y)
    pred_y = model.predict(test_x)
    accuracies.append(accuracy_score(test_y, pred_y))

plt.plot(accuracies)
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.title("Histogram of depth")

plt.show()
