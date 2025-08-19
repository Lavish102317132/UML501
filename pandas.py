import pandas as pd
#Q1
mydata={
    "Tid":[1,2,3,4,5,6,7,8,9,10],
    "Refund":["yes","no","no","yes","no","no","yes","no","no","no"],
    "Marital Status":["Single","Married","Single","Married","Divorced","Married","Divorced","Single","Married","Single"],
    "Taxable Amount":[125,100,70,120,95,60,220,85,75,90],
    "Cheat":["no","no","no","no","yes","no","no","yes","no","yes"]
}
df=pd.DataFrame(mydata)
#print(df)
#Q2
print(df.loc[0],df.loc[4],df.loc[7],df.loc[8])
#Q3
print(df.iloc[3:7])
print(df.iloc[4:8,2:4])
print(df.iloc[:,1:4])
#Q4
r=pd.read_csv('Iris.csv')
print(r.head())
#Q5
r.drop(4,axis=0,inplace=True)
r.drop(r.columns[2],axis=1,inplace=True)
print(r.head())
#Q6
employees={
    "EID":[101,102,103,104,105,106],
    "Name":["Alice","Bob","Charlie","Diana","Edward"],
    "Department":["HR","IT","IT","Marketing","Sales"],
    "Age":[29,34,41,28,38],
    "Salary":[50000,70000,65000,55000,60000],
    "Experience":[4,8,10,3,12],
    "Joining Date":["2020-03-15","2017-07-19","2013-06-01","2021-02-10","2010-11-25"],
    "Gender":["Female","Male","Male","Female","Male"],
    "Bonus":[5000,7000,6000,4500,5000],
    "Rating":[4.5,4.0,3.8,4.7,3.5]
}
df['Joining Date']=pd.to_datetime(df['Joining Date'])
df=pd.DataFrame(employees)
print(df)
print("Shape",df.shape(),"Info",df.info(),"Describe:",df.describe())
print(df.head(5),df.tail(3))
print("Average Salary:", df["Salary"].mean())
print("Total Bonus:", df["Bonus"].sum())
print("Youngest Employee Age:", df["Age"].min())
print("Highest Rating:", df["Rating"].max())
sortdec= df.sort_values(by="Salary", ascending=False)
print(sortdec)
df["Performance"]=pd.cut(
df["Rating"],
bins=[0, 4.0, 4.5, 5],
labels=["Average", "Good", "Excellent"]
)
print(df.isnull().sum())
df["Bonus"]=df["Bonus"].fillna(0)

bestemp=df[(df["Experience"]>5)&(df["Department"]=="IT")]
print(bestemp)
df["Tax"]=df["Salary"]*0.1
print(df) 
df.to_csv("employees.csv", index=False)