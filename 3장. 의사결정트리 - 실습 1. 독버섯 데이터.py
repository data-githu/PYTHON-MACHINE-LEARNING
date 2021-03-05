# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 22:55:59 2021

@author: Owner
"""

# ■ 3장. 의사결정트리 - 실습 1. 독버섯 데이터
import pandas as pd  # 데이터 전처리를 위해서 
import seaborn as sns # 시각화를 위해서 

df = pd.read_csv('c:\\data\\mushrooms.csv')

df = pd.get_dummies(df,drop_first=True)
print(df)
print(df.shape) # (8124, 96)
#%%

# get_dummies 함수를 이용해서 값의 종류에 따라 
# 전부 0 아니면 1로 변환함 

# DataFrame 확인
print(df.shape)  # (8124, 23)
print(df.info())  # 전부 object (문자)형으로 되어있음
print(df.describe())


#%%
# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = df.iloc[:,1:].to_numpy() 
y = df.iloc[:,0].to_numpy()   
print(X)
print(y)

print(df.shape)  # (8124, 96)
print(len(X))  # 8124
print(len(y))  # 8124


from sklearn import preprocessing 

X=preprocessing.StandardScaler().fit(X).transform(X)  
print(X)
#X=preprocessing.MinMaxScaler().fit(X).transform(X) 


from sklearn.model_selection import train_test_split 
                                                                                     
# 훈련 데이터 75, 테스트 데이터 25으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state = 10)

print(X_train.shape)   # (6093, 94)
print(y_train.shape)   # (6093,)



# 학습/예측(Training/Pradiction)

# sklearn 라이브러리에서 Decision Tree 분류 모형 가져오기
from sklearn import tree


#  의사결정트리 분류기를 생성 (criterion='entropy' 적용)
classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)


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
print(accuracy)  #  1.0   Decision tree 
                 #  0.9497   MultinomialNB
                 #  0.9615   GaussianNB
                 #  0.9350   BernoulliNB
