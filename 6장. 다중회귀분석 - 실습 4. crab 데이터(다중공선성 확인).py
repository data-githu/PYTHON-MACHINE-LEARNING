# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 23:12:50 2021

@author: Owner
"""

# ■  6장. 다중회귀분석 - 실습 4. crab 데이터(다중공선성 확인)
# 회귀분석에서 사용된 모형의 일부 독립변수가 다른 독립변수와의 상관정도가 아주 높아서 회귀분석 결과에 부정적 영향을 미치는 현상을 말합니다.

# 두 독립변수들끼리 서로에게 영향을 주고 있다면 둘 중 하나의 영향력을 검증할 때 다른 하나의 영향력이 약해집니다.

# 팽창계수가 보통은 10보다 큰 것을 골라내고 까다롭게 하려면 5보다 큰 것을 골라냅니다.
# 일평균음주량, 혈중알코올농도 둘다 팽창계수가 높게 나온다면 둘중에 하나를 빼고 아래와 같이 두번 회귀분석을 합니다.
# 학업성취도, 일평균음주량 --> 회귀분석
# 학업성취도, 혈중 알코올 농도 --> 회귀분석

import pandas as pd

# 1. 데이터 불러오기
df = pd.read_csv("c:\\data\\crab.csv")
print(df.head())
print(df.y.unique())

# 2. 다중회귀분석을 하고 종속변수에 영향을 주는 독립변수들이 무엇인지 확인하시오.
from statsmodels.formula.api import ols

model = ols('y ~ sat + weight + width', data=df)
result = model.fit()
print(result.summary()) # 0.514

# 3. 팽창계수를 확인합니다.
from statsmodels.stats.outliers_influence import variance_inflation_factor

print(model.exog_names) # 모델에서 분석한 독립변수들이 출력

# 위의 출력된 독립변수 중에 첫번째 컬럼의 팽창계수 확인
print(variance_inflation_factor(model.exog, 1)) # 1.1588368780857803
print(variance_inflation_factor(model.exog, 2)) # 4.8016794240392375
print(variance_inflation_factor(model.exog, 3)) # 4.688660343641888


# 4. 위의 팽창계수가 높은 두개의 독립변수를 각각 따로따로 이용해서 모델을 생성
model1 = ols('y ~ sat + width', data=df)
print(model1.fit().summary())

model2 = ols('y ~ sat + weight', data=df)
print(model2.fit().summary())


# 분석 결과 : weight 도 중요한 독립변수임을 확인
