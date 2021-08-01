import numpy as np
import pandas as pd
import pickle

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

iris_dataset = datasets.load_iris()
iris = pd.DataFrame(iris_dataset['data'] , columns = iris_dataset['feature_names'])

petalLength = iris['petal length (cm)']
petalWidth = iris['petal width (cm)']

X = np.column_stack((iris['sepal length (cm)'] , iris['sepal width (cm)'] , iris['petal length (cm)'] , iris['petal width (cm)']))
y = iris_dataset['target']

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3)

sv = svm.SVC(kernel = 'linear' , C = 1)
sv.fit(X_train , y_train)

pickle.dump(sv , open('iris.pkl' , 'wb'))