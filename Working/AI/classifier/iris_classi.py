#%%
from scipy.sparse.construct import random
from sklearn.datasets import load_iris
iris = load_iris()
#print(iris)
print(iris.__dir__())
iris.keys()
iris.data.shape 

# key 하나씩 쳐보기

#%%
import pandas as pd
pd.__version__
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df
# %%
iris_label = iris.target
iris_df['label'] = iris_label
iris_df

#%%
from sklearn.model_selection import train_test_split as ttst
X_train, X_test, Y_train, Y_test = ttst(iris.data,
                            iris_label,
                            test_size=0.2,
                            random_state=7)

print('X_train 개수:', len(X_train), 'X_test 개수:', len(X_test))

Y_train, Y_test
# %%
from sklearn.tree import DecisionTreeClassifier as DetreeCla
decision_tree=DetreeCla(random_state=32)
print(decision_tree._estimator_type)

# %%
decision_tree.fit(X_train,Y_train)

Y_pred=decision_tree.predict(X_test)
# %%
Y_pred
Y_test

from sklearn.metrics import accuracy_score as acsc
accuracy = acsc(Y_test,Y_pred)
print(accuracy)
# %%
# (1) 필요한 모듈 import
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# (2) 데이터 준비
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

# (3) train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(iris_data, 
                                                    iris_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

# (4) 모델 학습 및 예측
decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))
# %%
from sklearn.ensemble import RandomForestClassifier as RFCL
x_train, x_test, y_train, y_test = ttst(iris.data,
                            iris.target,
                            test_size= 0.2,
                            random_state=28)

random_forest = RFCL(random_state=32)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
print(classification_report(y_test, y_pred))

# %%
# How to use support vector mechine (SVM)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as ttst
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import accuracy_score as acsc
iris = load_iris()
x_train,x_test, y_train, y_test=ttst(iris.data,
                            iris.target,
                            test_size=0.2,
                            random_state=21)

svm_model=svm.SVC()
print(svm_model._estimator_type)
svm_model.fit(x_train, y_train)

y_pred= svm_model.predict(x_test)
accuracy = acsc(y_test,y_pred)
print(accuracy)
print(classification_report(y_test,y_pred))
# %%
# SGD Classifier(stochastic Gradient Descent Classifier)
from sklearn.linear_model import SGDClassifier
sgd_model=SGDClassifier()
print(sgd_model._estimator_type)

sgd_model.fit(x_train,y_train)
y_pred = sgd_model.predict(x_test)
from sklearn.metrics import accuracy_score as acsc
print('정확도', acsc(y_test,y_pred))

from sklearn.metrics import classification_report as clasireport
print(clasireport(y_test,y_pred))

# %%
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
print(logistic_model)
logistic_model.fit(x_train,y_train)
y_pred=logistic_model.predict(x_test)
print(acsc(y_pred,y_test))
print(clasireport(y_pred,y_test))
# %%
