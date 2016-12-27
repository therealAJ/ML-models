import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#get_ipython().magic('matplotlib inline')

customers = pd.read_csv('Ecommerce Customers')

#Insights
customers.head()
customers.info()
customers.describe()

#some visualizations
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers, kind='scatter')
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')

#Relationships between data
sns.pairplot(customers)

sns.lmplot(data=customers, x='Length of Membership', y='Yearly Amount Spent' )

#Lets Train

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 
               'Time on App',
               'Time on Website',
               'Length of Membership']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y test (True Value)')
plt.ylabel('Predicted Y')

#Evaluating the Model

from sklearn import metrics

print('MAE ', metrics.mean_absolute_error(y_test, predictions))
print('MSE ', metrics.mean_squared_error(y_test, predictions))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

metrics.explained_variance_score(y_test,predictions)

#Residuals
sns.distplot((y_test - predictions), bins=50)

cdf = pd.DataFrame(data=lm.coef_,index=X.columns,columns=['Coeff'])
print(cdf)

#plt.show()
#The Company should focus on their mobile app, since 'Time on App' has a coefficient of 38.59 as opposed to 'Time on Website' has a coefficient of 0.19