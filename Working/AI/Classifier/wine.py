#%%
# wine 데이터 로드
import numpy as np
from sklearn.datasets import load_wine
wine = load_wine() 
print(wine.keys())

# %%
# 성분확인
print(wine.target_names)
print(wine.feature_names)
print(wine.data)
print(wine.data.shape)




#%%
# Decision Tree 사용해 보기






#%%
# Random Forest 사용해 보기





#%%
# SVM 사용해 보기






#%%
# SGD Classifier 사용해 보기








#%%
# Logistic Regression 사용해 보기



