import pandas as pd 
import matplotlib.pyplot as plt
from bo_viz_lib import bo_scatter_matrix

df = pd.read_csv("../winequality-red.csv", sep=";")

# pd.plotting.scatter_matrix(df, figsize=(12,12))
# plt.show()

df['goodquality'] = [1 if x >= 6 else 0 for x in df['quality']]
df = df.drop('quality', axis=1)
bo_scatter_matrix(df, df['goodquality'])