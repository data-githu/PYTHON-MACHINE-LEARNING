# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:09:52 2021

@author: Owner
"""

# ■ 6장. 다중회귀분석 - 실습 2. 미국 대학 입학 데이터
import numpy as np
import statsmodels.api as sm # 회귀분석을 위해서
import statsmodels.formula.api as smf # 회귀분석을 위해서
import pandas as pd
from  sklearn.preprocessing   import  StandardScaler # 표준화를 위해서

# 1. data 불러오기
df = pd.read_csv("c:\\data\\sports.csv", engine='python', encoding='CP949') 
df.columns = ['stud_id', 'academic', 'sports', 'music', 'acceptance']

print(df)

# 2. 모델 생성하기
model = smf.ols(formula = 'acceptance ~ academic + sports + music' , data=df)
result = model.fit()
print(result.summary())

# 분석결과 : sport가 가장 많은 영향을 미친다는 결과가 나왔습니다.



# 표준화를 하고 분석을 다시 진행
import numpy as np
import statsmodels.api as sm # 회귀분석을 위해서
import statsmodels.formula.api as smf # 회귀분석을 위해서
import pandas as pd
from  sklearn.preprocessing   import  StandardScaler # 표준화를 위해서

# 1. data 불러오기
df = pd.read_csv("c:\\data\\sports.csv", engine='python', encoding='CP949') 
df.columns = ['stud_id', 'academic', 'sports', 'music', 'acceptance']

print(df)

# 2. 표준화하기
scaler = StandardScaler()
scaler.fit(df) # 표준화를 위해 df 데이터를 살펴본다.
df_scale = scaler.transform(df)


# 3. 판다스 데이터 프레임으로 구성합니다.
df_scale2 = pd.DataFrame(df_scale)
print(df_scale2.head())


# 4. 컬럼을 구성합니다.
df_scale2.columns = ['stud_id','academic','sports','music','acceptance'] # 컬럼명을 지정
print(df_scale2.head())


# 5. 모델 생성하기
model = smf.ols(formula = 'acceptance ~ academic + sports + music' , data = df_scale2)
result = model.fit()
print(result.summary())

# 분석결과  : 학과점수가 체육점수보다 더 영향력이 큰 독립변수로 나타나고 있습니다.
