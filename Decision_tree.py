#Decision  Tree Classification

#importing required libraries

import warnings
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier


plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")


#data extraction and cleaning
columns=['col'+str(i) for i in range(23)]
data= pd.read_csv('C:/Users/Mayank Sharma/Desktop/Semester 4/Data Mining And Warehousing/Assignments/computeIII/agaricus_lepiota.csv',names=columns)
X =data
y = X['col0']

for i in range(len(y)):
    if y[i]=='p':
        y[i]=0
    else:
        y[i]=1        
        

X.drop(['col0'], axis=1, inplace=True)


col=X.columns


#data preprocessing 


one_hot_encoded = pd.get_dummies(X[col])
X = pd.concat([X, one_hot_encoded], axis=1)
X = X.drop(col, axis=1)


X_total= np.asarray(X)
y_total=np.asarray(y)
y_total=y_total.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.30, random_state=35)



#Dicision tree classification
dt_sklearn = DecisionTreeClassifier(criterion = 'entropy', random_state = 35)

dt_sklearn.fit(X_train, y_train)


#visualization
plt.figure(figsize = (16, 9))
plot_tree(dt_sklearn, filled=True)
plt.show()


dt_sklearn_pred_test = dt_sklearn.predict(X_test)
dt_sklearn_pred_train = dt_sklearn.predict(X_train)



Accuracy_train = accuracy_score(dt_sklearn_pred_train, y_train)
Accuracy_test = accuracy_score(dt_sklearn_pred_test, y_test)
print(f"Training Accuracy is {Accuracy_train}")
print(f"Test Accuracy is {Accuracy_test}")






#Task-2  Information gain for each of atrribute

print("Information gain for each of atrribute\n")
for i,info_gain in enumerate(dt_sklearn.feature_importances_):
    print(f'col{i} : ',info_gain)








