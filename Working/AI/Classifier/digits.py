#%%
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.keys()) 
print(digits.feature_names)
print(np.array(digits.feature_names).shape)
#print(digits.DESCR)

# %%
import matplotlib.pyplot as plt
#%matplotlib inline
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(digits.data[i].reshape(8,8),cmap='gray')
    plt.axis('off')
plt.show() # 저 위에 % 도 그렇고 얘도 그렇고 역할이 없다.

digits_label = digits.target
print(digits_label.shape)
digits_label[:20]


# %%
new_label = [3 if i ==3 else 0 for i in digits_label]
new_label[:20]
from sklearn.model_selection import train_test_split as ttst
from sklearn.tree import DecisionTreeClassifier as DeTreeCla
from sklearn.metrics import accuracy_score as acsc
from sklearn.metrics import classification_report as clareport
x_train,x_test,y_train,y_test = ttst(digits.data,
                            new_label,
                            test_size=0.2,
                            random_state=21)
#Decision Tree 사용해 보기

Tree=DeTreeCla()
Tree.fit(x_train, y_train)
y_pred=Tree.predict(x_test)
accuracy = acsc(y_pred,y_test)
print(accuracy)
print(clareport(y_pred,y_test))
y_fake = [0]*len(y_pred)
print(acsc(y_fake,y_test))
print(clareport(y_fake,y_test))

# confusion matrix 오차행렬
from sklearn.metrics import confusion_matrix as confmatrix
print(confmatrix(y_test,y_pred))
print(confmatrix(y_test,y_fake))



# %%
# Random Forest 사용해 보기











#%%
# SVM 사용해 보기




#%%
# SGD Classifier 사용해 보기




#%%
# Logistic Regression 사용해 보기