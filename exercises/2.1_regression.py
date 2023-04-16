import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Use linear regression on wine dataset to predict how much alcohol a wine has; i.e. y=alcohol and test performance with RSME with a 80/20 split


df = pd.read_csv('data/winequality-red.csv', sep=";")

# Remove quality, add good quality
df['good_quality'] = [1 if x >= 6 else 0 for x in df['quality']]
df = df.drop('quality', axis=1)


X = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "good_quality"]]
X = X.to_numpy()
y = df['alcohol']
y = y.to_numpy()

# 80/20 split
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_x, train_y)

# Predict on the test data
pred_y = model.predict(test_x)

# Calcualte RMSE
rmse = mean_squared_error(test_y, pred_y, squared=False)

print(rmse)  # shoud be 0.595


# Plot predictions against real y
plt.scatter(pred_y, test_y)
plt.xlabel("Predicted alcohol")
plt.ylabel("True alcohol")
min = min([pred_y.min(), test_y.min()])
max = max([pred_y.max(), test_y.max()])
plt.axline((min, min), (max, max), color="black")
plt.show()
