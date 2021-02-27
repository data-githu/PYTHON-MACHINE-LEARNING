# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:32:15 2021

@author: Owner
"""

# ■ 2장. 나이브 베이즈 - 실습 1. iris 데이터
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas  as pd


# 데이터 준비
col_names = ['sepal-length', 'sepal-width','petal-length', 'petal-width','Class']

# csv 파일에서 DataFrame을 생성
df = pd.read_csv('c:\\data\\iris2.csv', encoding='UTF-8', header=None, names=col_names)
print(df)


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, :-1].to_numpy()
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
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit( X_train, y_train )

# 예측
y_pred= model.predict(X_test)
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
print(accuracy) # 0.7333

#%%

# 문제7. 위의 나이브베이즈 모델의 성능을 더 올리시오.
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas  as pd


# 데이터 준비
col_names = ['sepal-length', 'sepal-width','petal-length', 'petal-width','Class']

# csv 파일에서 DataFrame을 생성
df = pd.read_csv('c:\\data\\iris2.csv', encoding='UTF-8', header=None, names=col_names)
print(df)


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, :-1].to_numpy()
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
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

model = GaussianNB() # Gaussian Naive Bayes 모델 선택 - 연속형 자료
model.fit( X_train, y_train )

# 예측
y_pred= model.predict(X_test)
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
