# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:57:39 2021

@author: Owner
"""

# ■ 2장. 나이브 베이즈 - 실습 3. wine 데이터

# 문제10. 와인데이터를 나이브 베이즈 모델을 사용해서 정확도를 확인하시오
import pandas as pd # 데이터 전처리
import seaborn as sns # 시각화

df = pd.read_csv("c:\\data\\wine.csv") # StringAsFactor = T 부분 없음

# DataFrame 확인
print(df.shape) # 569행 32열
print(df.info()) # R의 str(df)와 동일 (데이터의 구조확인)
print(df.describe()) # R의 summary(df)와 동일 (요약 통계정보)

# 행을 선택하는 방법 emp[행][열] -> emp[조건][컬럼명]
print(df.iloc[0:5, ])  # 0-4번째 행 추출/ df.iloc[행번호,열번호]
print(df.iloc[-5: ,]) # 끝에서 5번째 행부터 끝까지 추출

# 열을 선택하는 방법 emp[행][열] -> emp[조건][c("ename","sal")]
print(df.iloc[ :, [0,1] ]) 
print(df.iloc[ :, : ]) 


# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:, 1:].to_numpy() # df 데이터프레임의 2번째 열부터 끝까지를 넘파이 array 로 변환
print(X)

y = df['Type'].to_numpy()   
print(y)


print(df.shape)

# 데이터 정규화를 수행한다.              
from sklearn import preprocessing 
X=preprocessing.StandardScaler().fit(X).transform(X)
#X=preprocessing.MinMaxScaler().fit(X).transform(X)
print(X)


from sklearn.model_selection import train_test_split 
                                                                
                     
# 훈련 데이터 70, 테스트 데이터 30으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.1, random_state = 10)
# random_state = 10 은 seed 값을 설정하는 부분 (동일한 정확도를 보기 위해서)
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
# model = BernoulliNB()

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
print(accuracy) # 0.8888888
