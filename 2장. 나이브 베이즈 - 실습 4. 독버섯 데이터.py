# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:46:59 2021

@author: Owner
"""

# ■ 2장. 나이브 베이즈 - 실습 4. 독버섯 데이터

# 문제12. 독버섯을 나이브 베이즈로 분류하시오 !
import pandas as pd  # 데이터 전처리를 위해서 
import seaborn as sns # 시각화를 위해서 

df = pd.read_csv("c:\\data\\mushrooms.csv") 

# DataFrame 확인
print(df.shape) # (8124, 23)
print(df.info()) # 모두 문자형

# get_dummies 함수를 이용해서 값을 0과 1로 변환
df = pd.get_dummies(df)
print(df.shape)  # (8124, 23)
print(df)
print(df.shape) # (8124, 119)

#%%

# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:,2:].to_numpy() 
y = df.iloc[:,1].to_numpy()   
print(X)
print(y)

print(df.shape)  # (8124, 119)
print(len(X))  # 8124
print(len(y))  # 8124


#%%
# from sklearn import preprocessing 

#X=preprocessing.StandardScaler().fit(X).transform(X) 
#X=preprocessing.MinMaxScaler().fit(X).transform(X) 

from sklearn.model_selection import train_test_split 
                                                                
                     
# 훈련 데이터 75, 테스트 데이터 25으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state = 10)

print(X_train.shape) # (6093, 21)
print(y_train.shape) 


#%%
# 스케일링(z-score 표준화 수행 결과 확인)
# for col in range(4):
#    print(f'평균 = {X_train[:, col].mean()}, 표준편차= {X_train[:, col].std()}')
    
#for col in range(4):
#   print(f'평균 = {X_test[:, col].mean()}, 표준편차= {X_test[:, col].std()}')



# 학습/예측(Training/Pradiction)
from sklearn.naive_bayes import MultinomialNB


# 나이브베이즈 분류기를 생성
classifier = MultinomialNB() 

# 분류기 학습
classifier.fit(X_train, y_train)

# 예측
y_pred= classifier.predict(X_test)
print(y_pred)

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
print(accuracy)  #  0.9497

# 문제13. 독버섯 분류 나이브 베이즈 모델의 정확도를 아래와 같이 0.99 로 만드는 laplace 값을 알아내시오 !
import pandas as pd  # 데이터 전처리를 위해서 
import seaborn as sns # 시각화를 위해서 

df = pd.read_csv("c:\\data\\mushrooms.csv") 

# DataFrame 확인
print(df.shape) # (8124, 23)
print(df.info()) # 모두 문자형

# get_dummies 함수를 이용해서 값을 0과 1로 변환
df = pd.get_dummies(df)

# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:,2:].to_numpy() 
y = df.iloc[:,1].to_numpy()   
print(X)
print(y)


from sklearn.model_selection import train_test_split 
                                                                
                     
# 훈련 데이터 75, 테스트 데이터 25으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state = 10)



# 학습/예측(Training/Pradiction)
from sklearn.naive_bayes import GaussianNB


# 나이브베이즈 분류기를 생성
classifier = GaussianNB(var_smoothing=0.004)


# 분류기 학습
classifier.fit(X_train, y_train)

# 예측
y_pred= classifier.predict(X_test)
print(y_pred)

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
print(accuracy)  #  0.9916


# %%
# 최적의 laplace 값 찾는 방법
import  numpy  as np

errors = []
for i in np.arange(0.001, 0.01 , 0.001):
    nb = GaussianNB(var_smoothing=i)
    nb.fit(X_train, y_train)
    pred_i = nb.predict(X_test)
    errors.append(np.mean(pred_i != y_test))
print(errors)

for k, i  in  zip(np.arange(0.001, 0.01 , 0.001),errors):
    print (k, '--->', i)
    
    
import matplotlib.pyplot as plt

plt.plot(np.arange(0.001, 0.01 , 0.001), errors, marker='o')
plt.title('Mean error with laplace-Value')
plt.xlabel('laplace-value')
plt.ylabel('mean error')
plt.show() # 0.004 선택
