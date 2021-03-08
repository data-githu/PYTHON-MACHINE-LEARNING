# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:30:15 2021

@author: Owner
"""

# ■ 4장. 회귀분석 - 실습 1. 자동차 연비 데이터 (단순회귀)
# 1970년대 후반과 1980년대 초반의 자동차 연비를 예측하는 모델을 만듭니다. 이 기간에 출시된 자동차 정보를 모델에 제공하였습니다. 이 정보에는 실린더 수 , 배기량, 마력, 공차 중량 같은 속성이 포함되어 있는 데이터 입니다.

### 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#[Step 1] 데이터 준비 - read_csv() 함수로 자동차 연비 데이터셋 가져오기

# CSV 파일을 데이터프레임으로 변환
df = pd.read_csv('c:\\data\\auto-mpg.csv', header=None)

# 열 이름 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
 'acceleration','model year','origin','name']

# 데이터 살펴보기
print(df.head())
print('\n')

# IPython 디스플레이 설정 - 출력할 열의 개수 한도 늘리기
pd.set_option('display.max_columns', 10)
print(df.head())
print('\n')


#[Step 2] 데이터 탐색
# 데이터 자료형 확인
print(df.info())
print('\n')

# 데이터 통계 요약정보 확인
print(df.describe())
print('\n')

# 설명 : horsepower 가 빠짐. 통계요약정보를 출력하려면 슛자형 데이터여야 합니다.
# mpg는 mile per gallon 의 약자로 영국과 미국에서는 한국과는 달리 갤런당 마일 단위로 연비표시
# 한국은 리터당 킬로미터 단위로 표시
# mpg 열을 한국에서 사용하는 km/l 로 변환해줘야합니다.
# 1갤런이 3.78541 이고 1마일이 1.60934 입니다.
# 그렇다면 1 mpg는?
# print(1.60934/3.78541) # 0.425km/l


# horsepower 열의 자료형 변경 (문자열 ->숫자)
print(df['horsepower'].unique()) # horsepower 열의 고유값 확인
print('\n')

df['horsepower'].replace('?', np.nan, inplace=True) # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True) # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float') # 문자열을 실수형으로 변환
print(df.describe()) # 데이터 통계 요약정보 확인
print('\n')


#[Step 3] 속성(feature 또는 variable) 선택

# 분석에 활용할 열(속성)을 선택 (연비, 실린더, 출력, 중량)
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]
print(ndf.head())
print('\n')

### 종속 변수 Y인 "연비(mpg)"와 다른 변수 간의 선형관계를 그래프(산점도)로 확인
# Matplotlib으로 산점도 그리기
ndf.plot(kind='scatter', x='weight', y='mpg', c='coral', s=10, figsize=(10, 5))
plt.show()
plt.close()

# seaborn으로 산점도 그리기
fig = plt.figure(figsize=(10, 5)) # 전체 그림판 가로 10, 세로 5로 설정
ax1 = fig.add_subplot(1, 2, 1) # 첫번째 그림판 영역
ax2 = fig.add_subplot(1, 2, 2) # 두번째 그림판 영역
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax1) # 회귀선 표시
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax2, fit_reg=False) #회귀선 미표시
plt.show()
plt.close()


# seaborn 조인트 그래프 - 산점도, 히스토그램
sns.jointplot(x='weight', y='mpg', data=ndf) # 회귀선 없음
sns.jointplot(x='weight', y='mpg', kind='reg', data=ndf) # 회귀선 표시
plt.show()
plt.close()

# seaborn pariplot으로 두 변수 간의 모든 경우의 수 그리기
sns.pairplot(ndf)
plt.show()
plt.close()

#Step 4: 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)

# 속성(변수) 선택
X=ndf[['weight']] #독립 변수 X
y=ndf['mpg'] #종속 변수 Y

# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, #독립 변수
 y, #종속 변수
 test_size=0.3, #검증 30%
 random_state=10) #랜덤 추출 값

print('train data 개수: ', len(X_train))
print('test data 개수: ', len(X_test))


# Step 5: 단순회귀분석 모형 - sklearn 사용

# sklearn 라이브러리에서 선형회귀분석 모듈 가져오기
from sklearn.linear_model import LinearRegression

# 단순회귀분석 모형 객체 생성
lr = LinearRegression()

# train data를 가지고 모형 학습
lr.fit(X_train, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
r_square = lr.score(X_test, y_test)
print(r_square)
print('\n')

# 회귀식의 기울기
print('기울기 a: ', lr.coef_)
print('\n')

# 회귀식의 y절편
print('y절편 b', lr.intercept_)
print('\n')

# 모형에 전체 X 데이터를 입력하여 예측한 값 y_hat을 실제 값 y와 비교
y_hat = lr.predict(X)
plt.figure(figsize=(10, 5))
ax1 = sns.distplot(y, hist=False, label="y")
ax2 = sns.distplot(y_hat, hist=False, label="y_hat", ax=ax1)
plt.show()
plt.close()
