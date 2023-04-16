import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/sa_heart.csv", sep=",")

# Number of columns
print("columns: ", len(df.columns))  # 12

# Number of rows
print("rows: ", len(df))  # 1599

# Stats
print(df.describe())

# ---
print(df)

# Visualise the data
sns.pairplot(df, hue="famhist")
plt.show()

# One-hot-encode famhist kolonnen
dummies = pd.get_dummies(df['famhist'])
df = df.join(dummies)
df = df.drop('famhist', axis=1)  # Fjern originale tabel
