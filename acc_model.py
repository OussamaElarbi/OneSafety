import pandas as pd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
'''activity nominal, {
                Walking=>1
                Jogging=>2
                Sitting=>3
                Standing=>4
                LyingDown =>5
                Stairs=>6
                Downstairs=>7 }'''
#collect the features and the target class
iris=load_iris()
# Clean the data from Nan ,missing and duplicate values
#dataset = dataset.dropna(axis=0, how='any')
x=iris.data
y=iris.target
'''
Use train/test split with different random_state values
'''
x_train , x_test , y_train, y_test = train_test_split(x,y,test_size=0.35,random_state=4)
# Check classification accuracy of Decision tree with
print('Decision tres model')
clf = tree.DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
# Testing accuracy
print ('Testing accuracy: ',round(metrics.accuracy_score(y_test,y_pred),4)*100,'%')
# Training accuracy
y_pred = clf.predict(x_train)
print('Training accuracy: ',round(metrics.accuracy_score(y_train,y_pred),4)*100,'%')
print()
print('K-nearest neighbor (KNN) model')
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
y_pred= knn.predict(x_test)
# Testing accuracy
print ('Testing accuracy: ',round(metrics.accuracy_score(y_test,y_pred),4)*100,'%')
# Training accuracy
y_pred = knn.predict(x_train)
print('Training accuracy: ',round(metrics.accuracy_score(y_train,y_pred),4)*100,'%')
#Try k=1 to k=35 for finding the best K
k_range=list(range(1,35))
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
plt.plot(k_range,scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing accuracy')
#plt.show()



# Using K Fold Cross validation for
# 1)Parameter tuning
# 2)Model selection
# 3)Feature Selection
knn = KNeighborsClassifier(n_neighbors=5)
tree.DecisionTreeClassifier()
scores = cross_val_score(knn,x,y,cv=5,scoring='accuracy')
print(scores)


