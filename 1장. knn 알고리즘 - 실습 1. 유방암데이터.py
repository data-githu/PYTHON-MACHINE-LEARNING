# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 23:26:32 2021

@author: Owner
"""


# ■ 1장. knn 알고리즘 - 실습 1. 유방암 데이터
import pandas as pd # 데이터 전처리
import seaborn as sns # 시각화

df = pd.read_csv("c:\\data\\wisc_bc_data.csv") # StringAsFactor = T 부분 없음

# 1. DataFrame
print(df.shape) # 569행 32열
print(df.info()) # R의 str(df)와 동일 (데이터의 구조확인)
print(df.describe()) # R의 summary(df)와 동일 (요약 통계정보)


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 2:].to_numpy() # df 데이터프레임의 2번째 열부터 끝까지를 넘파이 array 로 변환
print(X)

y = df['diagnosis'].to_numpy()   
print(y)


# 2. 데이터 정규화를 수행한다.              
from sklearn import preprocessing 
X=preprocessing.StandardScaler().fit(X).transform(X)
print(X)

  
# 3. 훈련 데이터 70, 테스트 데이터 30으로 나눈다. 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3, random_state = 10)
# random_state = 10 은 seed 값을 설정하는 부분 (동일한 정확도를 보기 위해서)
print(X_train.shape) 
print(y_train.shape) 



# 4. 학습/예측(Training/Pradiction)
from sklearn.neighbors import KNeighborsClassifier

# k-NN 분류기를 생성
classifier = KNeighborsClassifier(n_neighbors=5)

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
print(accuracy) # 0.9649


#%%

# 문제1. 위의 코드에서 적절한 k값을 알아내는 for 문을 구현하세요.
import  numpy  as np

errors = []
for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    errors.append(np.mean(pred_i != y_test))
print(errors)

for k,i in enumerate(errors):
    print(k,i)

#%%

# 문제2. 위에서 알아낸 가장 에러가 낮은 k값은 7,8,9,10 이였습니다. 그러면 k=7 일때 정확도를 확인하시오.
import pandas as pd # 데이터 전처리
import seaborn as sns # 시각화

df = pd.read_csv("c:\\data\\wisc_bc_data.csv") # StringAsFactor = T 부분 없음

# 1. DataFrame 확인
print(df.shape) # 569행 32열
print(df.info()) # R의 str(df)와 동일 (데이터의 구조확인)
print(df.describe()) # R의 summary(df)와 동일 (요약 통계정보)


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 2:].to_numpy() 
print(X)

y = df['diagnosis'].to_numpy()   
print(y)



# 2. 데이터 정규화를 수행한다.              
from sklearn import preprocessing 
X=preprocessing.StandardScaler().fit(X).transform(X)
print(X)

# 3. 훈련 데이터 70, 테스트 데이터 30으로 나눈다. 
from sklearn.model_selection import train_test_split 
                                                                
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3, random_state = 10)
# random_state = 10 은 seed 값을 설정하는 부분 (동일한 정확도를 보기 위해서)
print(X_train.shape) 
print(y_train.shape) 


# 스케일링(z-score 표준화 수행 결과 확인)
for col in range(4):
    print(f'평균 = {X_train[:, col].mean()}, 표준편차= {X_train[:, col].std()}')
    
for col in range(4):
    print(f'평균 = {X_test[:, col].mean()}, 표준편차= {X_test[:, col].std()}')


# 4. 학습/예측(Training/Pradiction)
from sklearn.neighbors import KNeighborsClassifier

# k-NN 분류기를 생성
classifier = KNeighborsClassifier(n_neighbors=7)

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
print(accuracy) # 0.9707

# 설명 : 0.97 의 정확도가 나옵니다. 의료 데이터이므로 정확도가 아주 높게 나와야 합니다. 그런데 정확도가 100% 가 나오면 좋겠는데 100%의 정확도가 나오기 어려우므로 FN 을 0으로 만들면 정확도가 100%가 아니라도 사용이 가능합니다.
# (FALSE NEGATIVE : 암환자를 정상환자로 잘못 예측했다.)

#%%

# 문제3. 지금 방금했던 시각화는 k값이 변경될 때 마다 오류가 어떻게 되는지 2차원 그래프로 시각화 한 것이고 이번에는 k값이 변경될 때마다 FN 값과 정확도가 어떻게 되는지 확인을 해야해보세요.

import matplotlib.pyplot as plt

plt.plot(range(1, 31), errors, marker='o')
plt.title('Mean error with K-Value')
plt.xlabel('k-value')
plt.ylabel('mean error')
plt.show()


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np


acclist = []
err_list = []
fn_list = []

for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    tn, fp,fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel() # ravel() 이원교차표 값들 사용가능
    fn_list.append(fn)
    acclist.append(accuracy_score(y_test, y_pred))
    err_list.append(np.mean(y_pred != y_test))

    print(f'k : {i} , acc : {accuracy_score(y_test, y_pred)} , FN : {fn}')

plt.figure(figsize=(12,6))
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=1, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)

# k값이 변경될 때마다 정확도가 어떻게 되는지 시각화
plt.subplot(131)
plt.plot(acclist,color='blue', marker='o', markerfacecolor='red')
plt.title('Accuracy', size=15)
plt.xlabel("k value")
plt.ylabel('Accuracy')

# k값이 변경될 때마다 오류가 어떻게 되는지 시각화
plt.subplot(132)
plt.plot(err_list, color='red', marker='o', markerfacecolor='blue')
plt.title('Error', size=15)
plt.xlabel("k value")
plt.ylabel('error')

# k값이 변경될 때마다 FN이 어떻게 되는지 시각화
plt.subplot(133)
plt.plot(fn_list, color='green', marker='o', markerfacecolor='yellow')
plt.title('FN Value', size=15)
plt.xlabel("k value")
plt.ylabel('fn value')

plt.show()


#%%

# 문제4. min-max 정규화를 통해서 데이터를 전처리 한 후 정확도를 확인하시오.
import pandas as pd # 데이터 전처리
import seaborn as sns # 시각화

df = pd.read_csv("c:\\data\\wisc_bc_data.csv") # StringAsFactor = T 부분 없음

# 1. DataFrame 확인
print(df.shape) # 569행 32열
print(df.info()) # R의 str(df)와 동일 (데이터의 구조확인)
print(df.describe()) # R의 summary(df)와 동일 (요약 통계정보)


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 2:].to_numpy() # df 데이터프레임의 2번째 열부터 끝까지를 넘파이 array 로 변환
print(X)

y = df['diagnosis'].to_numpy()   
print(y)


# 2. 데이터 정규화를 수행한다.              
from sklearn import preprocessing 
# X=preprocessing.StandardScaler().fit(X).transform(X)
X=preprocessing.MinMaxScaler().fit(X).transform(X)
print(X)
                                                                
                     
# 3. 훈련 데이터 70, 테스트 데이터 30으로 나눈다. 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3, random_state = 10)
# random_state = 10 은 seed 값을 설정하는 부분 (동일한 정확도를 보기 위해서)
print(X_train.shape) 
print(y_train.shape) 


# 스케일링(z-score 표준화 수행 결과 확인)
for col in range(4):
    print(f'평균 = {X_train[:, col].mean()}, 표준편차= {X_train[:, col].std()}')
    
for col in range(4):
    print(f'평균 = {X_test[:, col].mean()}, 표준편차= {X_test[:, col].std()}')


# 4. 학습/예측(Training/Pradiction)
from sklearn.neighbors import KNeighborsClassifier

# k-NN 분류기를 생성
classifier = KNeighborsClassifier(n_neighbors=7)

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
print(accuracy) # 0.982
