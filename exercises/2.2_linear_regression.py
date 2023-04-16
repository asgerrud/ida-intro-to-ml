from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Brug lineær regression på på hjerte datasættet til at forudse hvor gammel en person er. Altså hvor y = age og test performance med RSME med et 80/20 split.
df = pd.read_csv("data/sa_heart.csv", sep=",")

# One-hot encode famhist column
dummies = pd.get_dummies(df['famhist'])
df = df.join(dummies)
df = df.drop('famhist', axis=1)

# X og y
X = df[["sbp", "tobacco", "ldl", "adiposity",
        "Present", "Absent", "typea", "obesity", "alcohol", "chd"]]
X = X.to_numpy()
y = df['age']
y = y.to_numpy()


# 80/20 split
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(train_x, train_y)

# Predict on the test data
pred_y = model.predict(test_x)

# Calcualte RMSE
rmse = mean_squared_error(test_y, pred_y, squared=False)

print(rmse)  # shoud be 9.59

# Plot predictions against real y
plt.scatter(pred_y, test_y)
plt.xlabel("Predicted alcohol")
plt.ylabel("True alcohol")
min = min([pred_y.min(), test_y.min()])
max = max([pred_y.max(), test_y.max()])
plt.axline((min, min), (max, max), color="black")
plt.show()


# Opgave 3
# Prøv med de forskellige kolonner og se om nogen kan udlades
