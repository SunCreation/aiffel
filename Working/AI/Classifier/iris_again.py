#%%
from sklearn.datasets import load_iris

iris = load_iris()

dir(iris)
iris.__dir__()
# %%
iris.keys()
# %%
type(iris)
# %%
iris_data = iris.data

# %%
print(iris_data)
# %%
print(iris_data.shape)
# %%
print(type(iris_data))
# %%
iris_data
# %%
print(iris_data)
# %%
iris_data[0]
# %%
print(iris_data[0])
# %%
