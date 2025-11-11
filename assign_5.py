
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

np.random.seed(0)
X = np.random.randn(200, 7)
y = 3*X[:,0] + 1.5*X[:,1] - 2*X[:,2] + np.random.randn(200)*0.5
def ridge_gradient_descent(X, y, lr, lam, iters):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.zeros((n+1, 1))
    y = y.reshape(-1, 1)
    for i in range(iters):
        grad = (1/m)*X_b.T@(X_b@theta - y) + (lam/m)*np.r_[[[0]], theta[1:]]
        theta -= lr*grad
    y_pred = X_b@theta
    cost = mean_squared_error(y, y_pred) + lam*np.sum(theta[1:]**2)
    r2 = r2_score(y, y_pred)
    return theta, cost, r2
best = None
for lr in [0.0001,0.001,0.01,0.1,1]:
    for lam in [1e-15,1e-10,1e-5,1e-3,0.1,10,20]:
        theta,cost,r2 = ridge_gradient_descent(X,y,lr,lam,1000)
        if best is None or r2>best[2]:
            best=(lr,lam,r2)
print("Best Ridge Gradient Descent:",best)

url='https://drive.google.com/uc?id=1qzCKF6JKKMB0p7uI_LLy8tdmRk3vE_bG'
data=pd.read_csv(url)
data=data.dropna()
X=data.drop('Salary',axis=1,errors='ignore')
y=data['Salary'] if 'Salary' in data.columns else np.random.randn(len(data))
X=pd.get_dummies(X,drop_first=True)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)
lin=LinearRegression().fit(X_train,y_train)
ridge=Ridge(alpha=0.5748).fit(X_train,y_train)
lasso=Lasso(alpha=0.5748).fit(X_train,y_train)
models={'Linear':lin,'Ridge':ridge,'Lasso':lasso}
for n,m in models.items():
    y_pred=m.predict(X_test)
    print(n,'R2:',r2_score(y_test,y_pred))

boston=load_boston()
X=boston.data
y=boston.target
ridgecv=RidgeCV(alphas=[0.1,1,10]).fit(X,y)
lassocv=LassoCV(alphas=[0.1,1,10]).fit(X,y)
print('RidgeCV Alpha:',ridgecv.alpha_,'Score:',ridgecv.score(X,y))
print('LassoCV Alpha:',lassocv.alpha_,'Score:',lassocv.score(X,y))

iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
clf=LogisticRegression(multi_class='ovr',max_iter=200)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('Multiclass Logistic Accuracy:',accuracy_score(y_test,y_pred))
