import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/winequality-red.csv", sep=";")

# Number of columns
print("columns: ", len(df.columns))  # 12

# Number of rows
print("rows: ", len(df))  # 1599

# Stats
print(df.describe())

# ---

# Add new column based on other column
df['good_quality'] = [1 if x >= 6 else 0 for x in df['quality']]

# Remove column (axis=1 is column)
df = df.drop('quality', axis=1)
print(df)

# Visualise the data
sns.pairplot(df, hue="good_quality", height=1.1)
plt.show()
