# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

titanic_data = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')

# titanic_data = titanic_data[['Survived','Pclass','Sex','SibSp','Parch','Cabin','Ticket']]
# titanic_test = titanic_test[['Pclass','Sex','SibSp','Parch','Cabin','Ticket']]

nums = []
for tick in titanic_data['Ticket']:
    try:
        tick = int(tick)
        nums.append(tick)
    except:
        nums.append(tick)
        
titanic_data.drop(columns=['Ticket'])
titanic_data['Ticket'] = nums


# nums2 = []
# for tick2 in titanic_test['Ticket']:
#     try:
#         tick2 = int(tick2)
#         nums2.append(tick2)
#     except:
#         nums2.append(tick2)
        
# titanic_test.drop(columns=['Ticket'])
# titanic_test['Ticket'] = nums2


titanic_data = titanic_data[titanic_data['Ticket'].apply(lambda x: isinstance(x, int))]
# titanic_test = titanic_test[titanic_test['Ticket'].apply(lambda x: isinstance(x, int))]

titanic_data['Cabin'] = titanic_data['Cabin'].fillna(0)
# titanic_data['Cabin'] = titanic_data['Cabin'].fillna(0)


cabin1 = [] 
# cabin2 = []
for ca in titanic_data['Cabin']:
    if type(ca) is str:
        cabin1.append(1)
    else:
        cabin1.append(ca)
        
# for ca2 in titanic_test['Cabin']:
#     if type(ca2) is str:
#         cabin2.append(1)
#     else:
#         cabin2.append(ca2)
        
titanic_data['Cabin'] = cabin1
# titanic_test['Cabin'] = cabin2

titanic_data['Cabin'] = titanic_data['Cabin'].replace(['A'],'1')
titanic_data['Cabin'] = titanic_data['Cabin'].replace(['B'],'2')

titanic_test['Cabin'] = titanic_test['Cabin'].replace(['A'],'1')
titanic_test['Cabin'] = titanic_test['Cabin'].replace(['B'],'2')


titanic_data['Sex'] = titanic_data['Sex'].replace(['male'],'1')
titanic_data['Sex'] = titanic_data['Sex'].replace(['female'],'2')


titanic_test['Sex'] = titanic_data['Sex'].replace(['male'],'1')
titanic_test['Sex'] = titanic_data['Sex'].replace(['female'],'2')

# titanic_data = titanic_data.dropna()
# titanic_test = titanic_test.dropna()

# titanic_data= titanic_data.fillna(1.1) 
# titanic_test= titanic_test.fillna(1.1) 



# from sklearn import tree      
# from sklearn import ensemble  
# from sklearn.model_selection import cross_val_score, train_test_split

# features = ['Pclass','Sex','SibSp','Parch','Cabin','Ticket']

# X_train, X_test, Y_train, Y_test = train_test_split(titanic_data[features], titanic_data['Survived'], test_size=0.10, random_state=0)

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV

# clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
# clf.fit(X_train, Y_train)



# n_estimators = [int(x) for x in np.linspace(start = 50, stop = 100, num = 5)]
# max_features = ['auto', 'sqrt']
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]

# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#             }

# forest = RandomForestClassifier(random_state = 1, n_estimators = 300,max_depth = 2)
# rf_random = RandomizedSearchCV(estimator = forest, param_distributions = random_grid, cv = 3, random_state=1)
# modelF = forest.fit(X_train, Y_train)
# rf = rf_random.fit(X_train, Y_train)


# scores = cross_val_score(rf, X_train, Y_train, cv=5)

# score = rf.score(X_test, Y_test)
# y_pred = rf.predict(titanic_test)
# l = []
# l.append(y_pred)
# print(score)
# print(scores)
# for ll in l[0]:
#     print(ll)


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.cluster import KMeans


# x_x = titanic_data['Cabin']

# y_y = titanic_data['Survived']

# x_matrix = x_x.values.reshape(-1,1)


# reg = LinearRegression()
# reg.fit(x_matrix,y_y)

# print('score')
# print(reg.score(x_matrix,y_y))

x2 = titanic_data[['Cabin','Pclass','Parch','Sex','Ticket',]]

y2 = titanic_data['Survived']

reg = LinearRegression()


reg.fit(x2,y2)

print(reg.score(x2,y2))
