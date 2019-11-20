import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, Lasso, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics
from numpy import sqrt
from sklearn import naive_bayes
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from warnings import simplefilter

# simplefilter(action='ignore',category=FutureWarning)
from sklearn.preprocessing import StandardScaler

'''activity nominal, {
                Walking=>1
                Jogging=>2
                Sitting=>3
                Standing=>4
                LyingDown =>5
                Stairs=>6
                Downstairs=>7 }'''
# collect the features and the target class
iris = load_iris()
# Clean the data from Nan ,missing and duplicate values
# dataset = dataset.dropna(axis=0, how='any')
x = iris.data
y = iris.target
print(x.shape)
print(y.shape)
'''
Use train/test split with different random_state values
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=4)
# Check classification accuracy of Decision tree with
print('Decision tres model')
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
# Testing accuracy
print('Testing accuracy: ', round(metrics.accuracy_score(y_test, y_pred), 4) * 100, '%')
# Training accuracy
y_pred = clf.predict(x_train)
print('Training accuracy: ', round(metrics.accuracy_score(y_train, y_pred), 4) * 100, '%')
print()
print('K-nearest neighbor (KNN) model')
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
# Testing accuracy
print('Testing accuracy: ', round(metrics.accuracy_score(y_test, y_pred), 4) * 100, '%')
# Training accuracy
y_pred = knn.predict(x_train)
print('Training accuracy: ', round(metrics.accuracy_score(y_train, y_pred), 4) * 100, '%')
# Try k=1 to k=35 for finding the best K
k_range = list(range(1, 35))
scores = []
'''for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing accuracy')
plt.show()'''

# Using K Fold Cross validation for
# 1)Parameter tuning
# 2)Model selection
# 3)Feature Selection

'''# K Fold Cross Validation
kf= KFold(n_splits=5,shuffle=False).split(x,y)
# print the contents of each training and testing set
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))
    
# StratifiedKFold is preferred for classification problems
kf=StratifiedKFold(n_splits=5,shuffle=False).split(x,y)
# print the contents of each training and testing set
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))'''

# 1)Parameter tuning
# Compare the best value for KNN parameter K
'''k_scores=[]
k_range = list(range(1,31))
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    k_scores.append(cross_val_score(knn,x,y,cv=3,scoring='accuracy').mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN ')
plt.ylabel('Cross Validated Accuracy')
plt.show()

# Compare the best value number of K folds in cross validation
cv_scores = []
knn = KNeighborsClassifier(n_neighbors=5)
cv_range= list(range(2, 11))
for cv in cv_range:
    cv_scores.append(cross_val_score(knn, x, y, cv=cv, scoring='accuracy').mean())

plt.plot(cv_range, cv_scores)
plt.xlabel('Value of K folds for cross validation ')
plt.ylabel('Cross Validation Accuracy score')
plt.show()'''
print()
'''
# 2)Model selection
knn = KNeighborsClassifier(n_neighbors=5)
print('KNeighbors Classifier Accuracy: ',cross_val_score(knn,x,y,cv=3,scoring='accuracy').mean())
tree = tree.DecisionTreeClassifier()
print('Decision Tree Classifier Accuracy: ',cross_val_score(tree,x,y,cv=10,scoring='accuracy').mean())
log_reg = LogisticRegression(solver='liblinear',multi_class='auto')
print('Logistic Regression Accuracy: ',cross_val_score(log_reg,x,y,cv=10,scoring='accuracy').mean())
svc = svm.LinearSVC(max_iter=5000)
print('Linear SVC Accuracy: ',cross_val_score(svc,x,y,cv=10,scoring='accuracy').mean())
sgd = SGDClassifier()
print('SGD Classifier Accuracy: ',cross_val_score(sgd,x,y,cv=10,scoring='accuracy').mean())
gpc = GaussianProcessClassifier()
kernel = 1.0*RBF(1.0)
print('Gaussian Process Classifier Accuracy: ',cross_val_score(gpc,x,y,cv=10,scoring='accuracy').mean())


# 3)Feature Selection
# Adding /Removing unnecessary features to increase prediction performance
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, x, y, cv=3, scoring='neg_mean_squared_error')
mse = -scores
rmse = sqrt(mse)
rmse = rmse.mean()
print('All features accuracy score: ', rmse)
# iris dataset cols ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
df = pd.DataFrame(x, columns=iris.get('feature_names'))
df = df.drop(axis=0, columns='petal width (cm)')
scores = cross_val_score(knn, df, y, cv=3, scoring='neg_mean_squared_error')
mse = -scores
rmse = sqrt(mse)
rmse = rmse.mean()
print('3 features accuracy: ', rmse)

# Principle component analysis for dimensionality reduction and feature selection
x_std = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
# Apply the dimensionality reduction on x
y_sklearn = pca.fit_transform(x_std)
print(pca.explained_variance_ratio_)'''

# Exponential Moving Average , stddev
df = pd.DataFrame(x, columns=iris.get('feature_names'))
df['EMA'] = df['sepal length (cm)'].ewm(span=40, adjust=False).mean()
df['EMS'] = df['sepal length (cm)'].ewm(span=40, adjust=False).std()
df['EMV'] = df['sepal length (cm)'].ewm(span=40, adjust=False).var()
print(2/(len(df['sepal length (cm)'])+ 1))
df['Median'] = df['sepal length (cm)'].expanding(1).median()
print(df[['Median','sepal length (cm)']])
plt.figure(figsize=[15, 10])
plt.grid(True)
plt.plot(df['sepal length (cm)'], label='sepal length (cm)')
plt.plot(df['EMA'], label='EMA')
plt.plot(df['Median'], label='Median')
plt.plot(df['EMS'], label='EMS')
plt.plot(df['EMV'], label='EMV')
plt.legend(loc=2)
plt.show()
