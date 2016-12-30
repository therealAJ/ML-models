
# coding: utf-8

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

import seaborn as sns
iris = sns.load_dataset('iris')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

sns.pairplot(iris,hue='species',palette='Dark2')

setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True,shade_lowest=False)


# #### *Train Test Split*

from sklearn.model_selection import train_test_split

X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train,y_train)


# #### *Model Evaluation*

predictions = svc_model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# #### *Gridsearch* refinement

from sklearn.grid_search import GridSearchCV

param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}

grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,predictions))