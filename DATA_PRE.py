import pandas as pd
import numpy as np
dataset = pd.read_csv('C:/Users/csed.DESKTOP-RA25B6G/Desktop/archive/AWCustomers.csv')
#df1=dataset.copy()
#df1 = df1.fillna(np.average(dataset.HomeOwnerFlag))
#df1 = df1.fillna(np.average(dataset.NumberCarsOwned))

#df1 = df1.fillna(np.average(dataset.NumberChildrenAtHome))

#df1 = df1.fillna(np.average(dataset.TotalChildren))

#df1 = df1.fillna(np.average(dataset.YearlyIncome))

x = dataset.iloc[:, [0, 8, 13, 16, 17, 18, 19, 20, 21, 22]]
y=dataset.iloc[:,1]
print(pd.isna(y))
#y.dropna()
y.fillna(method='bfill')
print(y)