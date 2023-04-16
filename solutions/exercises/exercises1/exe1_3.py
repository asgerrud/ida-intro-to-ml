import pandas as pd 
import matplotlib.pyplot as plt
from bo_viz_lib import bo_scatter_matrix

df = pd.read_csv("../sa_heart.csv", sep=",")

bo_scatter_matrix(df, df['chd'])

famhist = pd.get_dummies(df['famhist'])
df = df.join(famhist)
df = df.drop('famhist', axis=1)
