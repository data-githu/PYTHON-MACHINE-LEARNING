# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 23:46:52 2021

@author: Owner
"""

# ■ 6장. 다중회귀분석 - 실습 1. 우주 왕복선의 폭파 원인 데이터
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

df = pd.read_csv("c:\\data\\challenger.csv", engine='python', encoding='CP949')
# print(df)

# 다중 회귀 분석 코드
model = smf.ols(formula = 'distress_ct ~ temperature + field_check_pressure + flight_num', data = df)
result = model.fit()
print(result.summary())

# 분석결과 설명 : O형링 파손에 영향을 주는 가장 큰 독립변수는 온도입니다.

# 문제20. statsmodels 패키지를 이용해서 방금 다중회귀 분석을 해보았는데 이번에는 중요한 독립변수 온도만을 이용해서 단순회귀 분석을 진행하시오.
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

df = pd.read_csv("c:\\data\\challenger.csv", engine='python', encoding='CP949')
# print(df)

# 단순 회귀 분석 코드
model = smf.ols(formula = 'distress_ct ~ temperature', data = df)
result = model.fit()
print(result.summary())

# y = 3.6984 - 0.0475*x1 (회귀식)
