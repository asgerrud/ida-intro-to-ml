from matplotlib.pyplot import *
import numpy as np
# Plot all attributes compared to each other

def __remove_binary_data__(df):
    tmp_list = []
    j = 0
    for i in df.dtypes: 
        if (i != object):
            tmp_list.append(j)
        j+=1
    return tmp_list


def bo_scatter_matrix(df, df_y): 
    X = df.to_numpy()
    cols = range(len(list(df.columns)))
    attributeNames = np.asarray(df.columns[cols])
    y = df_y.to_numpy()
    classNames = np.unique(y)
    C = len(classNames)
    df2 = df.drop(df_y.name, axis=1)
    Attributes = __remove_binary_data__(df2)
    NumAtr = len(Attributes)
    figure(figsize=(12,12))
    for m1 in range(NumAtr):
        for m2 in range(NumAtr):
            subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
            for c in classNames:
                if (m1 == m2): 
                    class_mask = (y==c)
                    hist(X[class_mask,Attributes[m2]])
                    if m1==NumAtr-1:
                        xlabel(attributeNames[Attributes[m2]])
                    else:
                        xticks([])
                    if m2==0:
                        ylabel(attributeNames[Attributes[m1]])
                    else:
                        yticks([])
                else: 
                    class_mask = (y==c)
                    plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
                    if m1==NumAtr-1:
                        xlabel(attributeNames[Attributes[m2]])
                    else:
                        xticks([])
                    if m2==0:
                        ylabel(attributeNames[Attributes[m1]])
                    else:
                        yticks([])
    legend(classNames)
    show()