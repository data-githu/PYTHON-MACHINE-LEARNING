# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:39:43 2021

@author: Owner
"""


■ 1장. knn 알고리즘 - 실습 2. iris 데이터

문제5. iris 데이터를 knn으로 분류하시오
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas  as pd


# 1. 데이터 준비
col_names = ['sepal-length', 'sepal-width','petal-length', 'petal-width','Class']

# csv 파일에서 DataFrame을 생성
df = pd.read_csv('c:\\data\\iris2.csv', encoding='UTF-8', header=None, names=col_names)
print(df)


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 0:-1].to_numpy()
print(X)

y = df['Class'].to_numpy()   
print(y)


print(df.shape)

from sklearn import preprocessing 
X=preprocessing.StandardScaler().fit(X).transform(X) 

from sklearn.model_selection import train_test_split 

# 훈련 데이터 70, 테스트 데이터 30으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3, random_state = 10)

print(X_train.shape) 
print(y_train.shape)


# 스케일링(z-score 표준화 수행 결과 확인)
for col in range(4):
    print(f'평균 = {X_train[:, col].mean()}, 표준편차= {X_train[:, col].std()}')
    
for col in range(4):
    print(f'평균 = {X_test[:, col].mean()}, 표준편차= {X_test[:, col].std()}')


# 학습/예측(Training/Pradiction)
from sklearn.neighbors import KNeighborsClassifier

# k-NN 분류기를 생성
classifier = KNeighborsClassifier(n_neighbors=10)

# 분류기 학습
classifier.fit(X_train, y_train)

# 예측
y_pred= classifier.predict(X_test)
print(y_pred)

# 모델 평가
from sklearn.metrics import confusion_matrix
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)    

# 이원 교차표 보는 코드 
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

# 정확도 확인하는 코드 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score( y_test, y_pred)
print(accuracy) # 1.0



문제6. iris 데이터에 대해서 가장 정확도가 좋은 k값을 지정해서 아이리스 데이터를 분류하는 knn 모델을 생성하는 전체 코드를 올리시오
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas  as pd


# 1. 데이터 준비
col_names = ['sepal-length', 'sepal-width','petal-length', 'petal-width','Class']

# csv 파일에서 DataFrame을 생성
df = pd.read_csv('c:\\data\\iris2.csv', encoding='UTF-8', header=None, names=col_names)
print(df)


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 0:-1].to_numpy()
print(X)

y = df['Class'].to_numpy()   
print(y)


print(df.shape)

from sklearn import preprocessing 
X=preprocessing.StandardScaler().fit(X).transform(X) 

from sklearn.model_selection import train_test_split 

# 훈련 데이터 70, 테스트 데이터 30으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3, random_state = 10)

print(X_train.shape) 
print(y_train.shape)


# 스케일링(z-score 표준화 수행 결과 확인)
for col in range(4):
    print(f'평균 = {X_train[:, col].mean()}, 표준편차= {X_train[:, col].std()}')
    
for col in range(4):
    print(f'평균 = {X_test[:, col].mean()}, 표준편차= {X_test[:, col].std()}')


# 학습/예측(Training/Pradiction)
from sklearn.neighbors import KNeighborsClassifier

# k-NN 분류기를 생성
classifier = KNeighborsClassifier(n_neighbors=15)

# 분류기 학습
classifier.fit(X_train, y_train)

# 예측
y_pred= classifier.predict(X_test)
print(y_pred)

# 모델 평가
from sklearn.metrics import confusion_matrix
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)    

# 이원 교차표 보는 코드 
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

# 정확도 확인하는 코드 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score( y_test, y_pred)
print(accuracy) # 1.0


#%%

import  numpy  as np

errors = []
for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    errors.append(np.mean(pred_i != y_test))
print(errors)


import matplotlib.pyplot as plt

plt.plot(range(1, 31), errors, marker='o')
plt.title('Mean error with K-Value')
plt.xlabel('k-value')
plt.ylabel('mean error')
plt.show() # 15 선택
