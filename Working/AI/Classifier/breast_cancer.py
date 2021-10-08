#%%
# 사용할 라이브러리 도입
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

breast_cancer = load_breast_cancer()
print(breast_cancer.keys())
print(breast_cancer.data.shape)
print(breast_cancer.target.shape)
print(breast_cancer.target_names)
print(breast_cancer.DESCR)


# %%



# %%
# Decision Tree 사용해 보기






#%%
# Random Forest 사용해 보기







#%%
# SVM 사용해 보기





#%%
# SGD Classifier 사용해 보기







#%%
# Logistic Regression 사용해 보기