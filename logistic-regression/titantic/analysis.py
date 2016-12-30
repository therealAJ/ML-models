
# coding: utf-8

# ## Titantic Survival Rate Analysis

# #### *Import Pandas, Numpy, and Visualization Libraries*

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# #### *Create Training DataFrame*

train_df = pd.read_csv('titanic_train.csv')

# #### *Quick Stats on the data*
train_df.head()
train_df.info()

sns.set_style('whitegrid')

# #### *Male vs Female survival count*

sns.countplot(data=train_df,x='Survived',hue='Sex',palette='RdBu_r')

# #### *By Passenger Class survival count*

sns.countplot(data=train_df,x='Survived',hue='Pclass')

# #### Age distribution of passengers

sns.distplot(train_df['Age'].dropna(),kde=False,bins=30)
sns.countplot(data=train_df,x='SibSp')

# #### *Fare Distribution*

train_df['Fare'].hist(bins=40,figsize=(10,4))
plt.xlabel('Fare Paid')
plt.ylabel('Number of Passengers')


#  ### *Time to remove missing data*

sns.heatmap(train_df.isnull(),yticklabels=False,cmap='viridis',cbar=False)

pclass_1_mean_age = train_df.groupby(['Pclass']).mean().iloc[0]['Age']
pclass_2_mean_age = train_df.groupby(['Pclass']).mean().iloc[1]['Age']
pclass_3_mean_age = train_df.groupby(['Pclass']).mean().iloc[2]['Age']

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return pclass_1_mean_age
        elif Pclass == 2:
            return pclass_2_mean_age
        else:
            return pclass_3_mean_age
    else:
        return Age

train_df['Age'] = train_df[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(train_df.isnull(),yticklabels=False,cmap='viridis',cbar=False)

train_df.drop('Cabin',axis=1,inplace=True)

train_df.dropna(inplace=True)

sns.heatmap(train_df.isnull(),yticklabels=False,cmap='viridis',cbar=False)

sex = pd.get_dummies(train_df['Sex'],drop_first=True)
embark = pd.get_dummies(train_df['Embarked'],drop_first=True)

train_df = pd.concat([train_df,sex,embark],axis=1)
train_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train_df.drop('PassengerId',axis=1,inplace=True)

# ### *Time to train our model*

y = train_df['Survived'].copy()
X = train_df.drop('Survived',axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predicitions = logmodel.predict(X_test)

# ### *Time to evaluate our model*

from sklearn.metrics import classification_report

print(classification_report(y_test,predicitions))

# #### Fooling around seeing predictions

from collections import OrderedDict

d = OrderedDict({'Pclass': 1,
                 'Age': 5,
                 'SibSp': 0,
                 'Parch': 0,
                 'Fare' : 51,
                 'male': 1,
                 'Q': 1,
                 'S': 0
                })

fake_df = pd.DataFrame(d,index=[0])
print(logmodel.predict_proba(fake_df))

