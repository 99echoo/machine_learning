#!/usr/bin/env python
# coding: utf-8

# In[11]:


### week3 problem 1 ######
#### 이름: 김동현####


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Darwin': #맥
        plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': #윈도우
        plt.rc('font', family='Malgun Gothic')
KAKAO = yf.download('035720.KS','2019-01-04','2023-01-05')
KAKAO['KAKAO_30MA'] = KAKAO['Close'].rolling(30).mean()
KAKAO['KAKAO_90MA'] = KAKAO['Close'].rolling(90).mean()
KAKAO['카카오주가'] = KAKAO['Close']
KAKAO.dropna(axis=0,inplace= True)
KAKAO[['카카오주가','KAKAO_30MA','KAKAO_90MA']].plot(title='카카오주가, 카카오주가30일, 90일 이동평균선')


# In[1]:


### week3 problem 1 ######
#### 이름: 김동현####

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import platform
if platform.system() == 'Darwin': #맥
        plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': #윈도우
        plt.rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False
stock = 'KAKAO'
code = '035720.KS'
start = '2019-01-04'
end = '2023-01-05'
data = yf.download(code,start,end)
data['Close_return']= data['Close'].pct_change()
data.dropna(axis=0,inplace=True)
sns.distplot(data['Close_return'],bins=100)
plt.xlabel('Daily return')
plt.ylabel('Number of days')
plt.title('Distribution of ' + stock + ' daily returns between ' + start + 'and ' + end)


# In[ ]:





# In[26]:


### week3 problem 1 ######
#### 이름: 김동현####

import pandas as pd

df1=pd.read_excel('C:/Users/user/Downloads/OneDrive_2023-03-13/saledata.xlsx')
df1['일자']=pd.to_datetime(df1['일자'])
df1.set_index('일자',inplace=True)
df_2=df1.groupby('상품').resample('M')['주문수량'].sum().round(0)
df_2


# In[ ]:




