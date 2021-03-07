# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 23:44:37 2021

@author: Owner
"""

# ■ 3장. 의사결정트리 - 실습 3. credit 데이터
# 문제17.  독일 credit 데이터를 이용해서 의사결정트리 모델을 생성하시오 !

import pandas as pd  # 데이터 전처리를 위해서 
import seaborn as sns # 시각화를 위해서 

df =  pd.read_csv('c:\\data\\credit.csv', encoding='UTF-8')

df = pd.get_dummies(df, drop_first=True)
df

X = df.iloc[:,0:35]
y = df.iloc[:,-1]
print(X)
print(y)

# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:,0:35].to_numpy() 
y = df.iloc[:,-1].to_numpy()   
print(X)  
print(y) 

from sklearn.model_selection import train_test_split 
                                                                                     
# 훈련 데이터 75, 테스트 데이터 25으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state = 10)

print(X_train.shape)   # (750, 35)
print(y_train.shape)   # (750,)

from sklearn import tree

#  의사결정트리 분류기를 생성 (criterion='entropy' 적용)
classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
#classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
# 메뉴얼 : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# 분류기 학습
classifier.fit(X_train, y_train)

# 특성 중요도
print(df.columns.values[0:35])
print("특성 중요도 : \n{}".format(classifier.feature_importances_))

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_cancer(model):
    n_features = df.shape[1]-1
    plt.barh(range(n_features),model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df.columns.values[0:35])
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1,n_features)

plot_feature_importances_cancer(classifier)
plt.show()

y_pred= classifier.predict(X_test)

# 작은 이원교차표
from sklearn.metrics import confusion_matrix
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)   

# 정밀도 , 재현율, f1 score 확인 
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

# 정확도 확인하는 코드 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score( y_test, y_pred)
print( accuracy) # 0.696

