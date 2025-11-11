import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

class ManualGaussianNB:
    def fit(self, X, y, epsilon=1e-9):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.means_ = np.zeros((len(self.classes_), n_features))
        self.vars_ = np.zeros((len(self.classes_), n_features))
        self.class_prior_ = np.zeros(len(self.classes_))
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = X_c.mean(axis=0)
            self.vars_[idx, :] = X_c.var(axis=0) + epsilon
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]
        return self

    def _gaussian_log_likelihood(self, class_idx, X):
        mean = self.means_[class_idx]
        var = self.vars_[class_idx]
        const = -0.5 * np.log(2.0 * np.pi * var)
        exponent = -((X - mean) ** 2) / (2.0 * var)
        return (const + exponent).sum(axis=1)

    def predict(self, X):
        log_probs = []
        for idx, c in enumerate(self.classes_):
            log_prior = np.log(self.class_prior_[idx])
            log_likelihood = self._gaussian_log_likelihood(idx, X)
            log_probs.append(log_prior + log_likelihood)
        log_probs = np.vstack(log_probs).T
        return self.classes_[np.argmax(log_probs, axis=1)]

manual_gnb = ManualGaussianNB().fit(X_train, y_train)
y_pred_manual = manual_gnb.predict(X_test)
acc_manual = accuracy_score(y_test, y_pred_manual)

sk_gnb = GaussianNB().fit(X_train, y_train)
y_pred_sk = sk_gnb.predict(X_test)
acc_sk = accuracy_score(y_test, y_pred_sk)

print("Gaussian Naive Bayes (Manual) accuracy on test set: {:.4f}".format(acc_manual))
print("Gaussian Naive Bayes (sklearn) accuracy on test set: {:.4f}".format(acc_sk))
print("\nClassification report (Manual GNB):\n", classification_report(y_test, y_pred_manual))
print("Confusion matrix (Manual GNB):\n", confusion_matrix(y_test, y_pred_manual))
print("\nClassification report (sklearn GNB):\n", classification_report(y_test, y_pred_sk))
print("Confusion matrix (sklearn GNB):\n", confusion_matrix(y_test, y_pred_sk))

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1, 31)), 'weights': ['uniform', 'distance']}
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
grid.fit(X_train, y_train)

best_params = grid.best_params_
best_score = grid.best_score_

print("\nGridSearchCV results for K-NN:")
print("Best params:", best_params)
print("Best cross-validation accuracy: {:.4f}".format(best_score))

best_knn = grid.best_estimator_
y_pred_knn = best_knn.predict(X_test)
acc_knn_test = accuracy_score(y_test, y_pred_knn)
print("Test set accuracy with best K-NN: {:.4f}".format(acc_knn_test))
print("\nClassification report (Best K-NN):\n", classification_report(y_test, y_pred_knn))
print("Confusion matrix (Best K-NN):\n", confusion_matrix(y_test, y_pred_knn))

results = pd.DataFrame(grid.cv_results_)[['param_n_neighbors','param_weights','mean_test_score','std_test_score','mean_train_score']]
results = results.sort_values(['param_weights','param_n_neighbors']).reset_index(drop=True)
results.to_csv('/mnt/data/gridsearch_knn_results.csv', index=False)
print("\nSaved GridSearchCV results to /mnt/data/gridsearch_knn_results.csv")
