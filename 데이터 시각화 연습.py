#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/user/Desktop/AICE자료/Clean_Dataset.csv")
df.drop([df.columns[0]],axis=1,inplace=True)
df


# In[ ]:





# In[5]:


df.describe()


# In[6]:


df.info()


# In[9]:


print(df['airline'].value_counts())
print(df['destination_city'].value_counts())
print(df['source_city'].value_counts())


# In[10]:


df.corr()


# In[14]:


df_eco = df[(df['class'] == 'Economy')]
df_eco.corr()


# In[15]:


pd.crosstab(df['source_city'],df['departure_time'])


# In[27]:


df1 = df.groupby(['source_city','departure_time']).mean()
df1
'''
그냥 묶는 방법은 pd.crosstab을 사용하면 되고 , 그룹으로 묶어서 무언가를 알고 싶을때는 groupby를 사용하면 된다.
'''


# In[29]:


days_left = df.groupby('days_left').mean()
days_left


# In[31]:


plt.figure()
plt.plot(days_left['price'])
plt.xlabel('Days_left')
plt.ylabel('Price')
plt.show()


# In[47]:


df2 = df.groupby(['airline']).mean()
airline = df2.index
plt.figure()
plt.bar(df2.index, df2['price'])
plt.show()
'''
bar chart는 인자로 x의 칼럼과 y값을 나타내는 값을 인자로 받는다,
'''


# In[58]:


df3 = df['airline'].value_counts()
plt.figure(figsize=(8,4))
plt.pie(df3,labels=df3.index,autopct='%.1f%%')
plt.show()
'''
범주변수 분포를 알고싶을때는 piechart를 사용한다.
'''


# In[4]:


import matplotlib.pyplot as plt
plt.hist(df['duration'],bins=10)
plt.xlabel('Duration')
plt.ylabel('Flight')
plt.show()
'''
수치변수의 분포를 알고싶은 경우에는 histogram을 사용한다
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[53]:


plt.figure(figsize=(8,4))
plt.scatter(x = df['duration'], y= df['price'])
plt.xlabel('duration')
plt.ylabel('price')
plt.show()


# In[55]:


heat = df_eco.corr()
plt.pcolor(heat)
plt.colorbar()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




