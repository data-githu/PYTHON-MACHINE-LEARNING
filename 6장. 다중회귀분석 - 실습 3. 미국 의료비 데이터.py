# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 23:11:55 2021

@author: Owner
"""

# ■ 6장. 다중회귀분석 - 실습 3. 미국 의료비 데이터
문제21. 미국 의료비 데이터를 가지고 다중 회귀분석을 진행하시오
import numpy as np
import statsmodels.api as sm # 회귀분석을 위해서
import statsmodels.formula.api as smf # 회귀분석을 위해서
import pandas as pd

# 1. data 불러오기
df = pd.read_csv("c:\\data\\insurance.csv", engine='python', encoding='CP949') 
print(df)


# 2. 모델 생성하기
model = smf.ols(formula = 'expenses ~ age + sex + bmi + children + smoker + region' , data = df)
result = model.fit()
print(result.summary())

# 분석결과 : sex[T.male] -131.3520 -> 남성은 여성에 비해 매년 의료비가 131달러 적게 들 거라고 예상이 가능
# 분석결과 : smoker[T.yes] 2.385e+04 -> 흡연자는 비흡연자보다 매년 의료비가 2.386*10^4 = 23,860 달러 더 든다.
# 결정계수 : 0.751
#%%
문제22. 비만인 사람은 의료비가 더 지출이 되는지 파생변수를 추가해서 확인하시오. bmi30 이라는 파생변수 bmi가 30이면 1 아니면 0 이라고 해서 컬럼을 추가하시오.
import numpy as np
import statsmodels.api as sm # 회귀분석을 위해서
import statsmodels.formula.api as smf # 회귀분석을 위해서
import pandas as pd

# 1. data 불러오기
df = pd.read_csv("c:\\data\\insurance.csv", engine='python', encoding='CP949') 
print(df)


# 2. 파생변수 만들기
def func_1(x):
    if x >= 30 :
        return 1
    else:
        return 0
    
df['bmi30'] = df['bmi'].apply(func_1)    
print(df)


문제23. bmi30 파생변수를 포함한 결정계수를 확인하시오.
# 3. 모델 생성하기
model = smf.ols(formula = 'expenses ~ age + sex + bmi + children + smoker + region + bmi30' , data = df)
result = model.fit()
print(result.summary())

# 분석결과 : 0.751 -> 0.756 으로 올라갔습니다. 
#%%
문제24. 비만이면 흡연까지 하면 의료비가 더 올라가는지 확인하시오.
import numpy as np
import statsmodels.api as sm # 회귀분석을 위해서
import statsmodels.formula.api as smf # 회귀분석을 위해서
import pandas as pd

# 1. data 불러오기
df = pd.read_csv("c:\\data\\insurance.csv", engine='python', encoding='CP949') 
print(df)


# 2. 파생변수 만들기
def func_1(x):
    if x >= 30 :
        return 1
    else:
        return 0
    
df['bmi30'] = df['bmi'].apply(func_1)    
print(df)

# 3. 모델 생성하기
model = smf.ols(formula = 'expenses ~ age + sex + bmi + children + smoker + region + bmi30 \
                + bmi30*smoker' , data = df)
result = model.fit()
print(result.summary()) # 0.864

# 분석 결과 : 비만이면서 흡연까지 하게 되면 연간 의료비가 19,790 달러 더 들거라 예상이 됩니다.

