#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
### 데이터셋 불러오기
df = pd.read_csv("C:/Users/user/Desktop/AICE자료/Clean_Dataset.csv")
df = df[:10000]

df

'''
airline = 항공사 이름 flight = 항공편 명 source_city = 출발 도시 departure_time 출발 시간 stops 환승장 수 arrival_time 도착시간
destination_city = 도착 도시 Class = 좌석 등급 duration = 비행 시간 Days_left 출발까지 남은시간 price = 가격
'''
## 구하고 싶은것은 가격 예측


# In[115]:


df.info()
#### object 데이터를 string으로 변환 해야하고, stops, class는 순서가 필요할지도 모른다.


# In[116]:


df.describe()


# In[30]:


###데이터 시각화
#1 항공사 별 가격 파악
import matplotlib.pyplot as plt
import seaborn as sns

airline_value = df.groupby(['airline']).mean()

plt.figure(figsize=(8,4))
plt.bar(airline_value.index, airline_value['price'])
plt.show()
'''
Air_india, Vistara가 높은것을 알 수 있음, AIr_Asia는 쌈
'''


# In[41]:


#2 출발시간, 경유에 따른 가격 확인
departure_time = df['departure_time'].value_counts()
departure_time
plt.figure(figsize=(8,4))
plt.pie(departure_time, labels = departure_time.index)
plt.show()


# In[42]:


stops = df['stops'].value_counts()
plt.figure(figsize=(8,4))
plt.pie(stops, labels = stops.index)
plt.show()


# In[54]:


##항공사 별 경유 분포를 알아보자
a = pd.crosstab(df['airline'],df['stops'])
a


# In[58]:


#3 남은 시간 별 가격
plt.scatter(x=df['days_left'],y=df['price'])
plt.show()
'''
일찍 살 수록 싼건 확실하다.
'''
airline = 항공사 이름 flight = 항공편 명 source_city = 출발 도시 departure_time 출발 시간 stops 환승장 수 arrival_time 도착시간
destination_city = 도착 도시 Class = 좌석 등급 duration = 비행 시간 Days_left 출발까지 남은시간 price = 가격


# In[2]:


###데이터 전처리
df = df.drop(['Unnamed: 0','flight'],axis=1)
df

'''
departure_time, stops, arrival_time, class는 순서가 필요한 범주라고 생각함
'''


# In[3]:


df


# In[4]:


import pandas as pd
import numpy as np
##LabelEncoding 과 원핫 인코딩
'''
airline, source_city, destination_city는 원핫 인코딩, departure_time, arrival_time, class,stops는 LabelEncoding할것이다.
'''

### label encoding
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
columns_to_encode = ['departure_time', 'arrival_time', 'class','stops']
for column in columns_to_encode:
    df[column] = Le.fit_transform(df[column])

###One-hot Encoding
df = pd.get_dummies(df, columns = ['airline','source_city','destination_city'])

###데이터셋 분리
X = df.drop(['price'],axis=1)
y= df['price']

##정규화
from sklearn.preprocessing import MinMaxScaler
Mn = MinMaxScaler()
X = Mn.fit_transform(X)

###학습과 테스트 데이터 분리

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

###학습 모델 적용


# In[18]:


#### 머신러닝 모델 생성
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
### LR이라는 학습 모델이 만들어진 것이다.

y_pred = LR.predict(X_test)

### 성능 평가 (MSE)
from sklearn.metrics import mean_squared_error, r2_score
r2_score = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
print(LR.score(X_train,y_train))
print(LR.score(X_test,y_test))
print('MSE 의 값은 :', mse)
print('R2_score의 값은 :', r2_score)

import matplotlib.pyplot as plt
y_pred = LR.predict(X_test)

plt.plot(X_test, y_test, 'bo', label='Actual')  # 실제 값 (파란색 점)
plt.plot(X_test, y_pred, 'ro', label='Predicted')  # 예측 값 (빨간색 점)
plt.xlabel('X_test')
plt.ylabel('y values')
plt.legend()
plt.title('Actual vs. Predicted (Linear Regression)')
plt.show()


# In[7]:


####Decision_tree 생성
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor()
DT.fit(X_train,y_train)

y_pred = DT.predict(X_test)

### 성능평가
from sklearn.metrics import mean_squared_error, r2_score
r2_score = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
print(DT.score(X_train,y_train))
print(DT.score(X_test,y_test))
print(mse)
print(r2_score)


# In[16]:


###RandomForest 생성
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X_train,y_train)

###GridSearchCv
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [10, 30, 50], # 트리의 개수 지정
    'max_depth': [None, 3, 5],
    'min_samples_split': [2, 4, 8],  # 노드를 분활하기 위해 필요한 최소 샘플 수 지정
    'min_samples_leaf': [1, 2, 4]  # 리프 노드가 되기 위해 필요한 최소 샘플 수 지정 
}

GE = GridSearchCV(estimator=RF, param_grid = param_grid, cv=5)
GE.fit(X_train,y_train)

y_pred = GE.predict(X_test)

### 성능평가
from sklearn.metrics import mean_squared_error, r2_score
r2_score = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
print(DT.score(X_train,y_train))
print(DT.score(X_test,y_test))
print(mse)
print(r2_score)


# In[15]:


##DNN
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

DNN = Sequential()
DNN.add(Dense(64,activation = 'relu', input_shape =(15,)))
DNN.add(Dense(32,activation= 'relu'))
DNN.add(Dropout(0.3))
DNN.add(Dense(32,activation= 'relu'))
DNN.add(Dropout(0.2))
DNN.add(Dense(1))

DNN.compile(optimizer='adam',loss='mean_squared_error',metrics = ['mae'])

es= EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1,restore_best_weights=True)
history = DNN.fit(X_train,y_train,epochs=1000,batch_size=128,
                   verbose=1,validation_data=(X_test,y_test),callbacks=[es])


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


("'departure_time','arrival_time','class'")


# In[ ]:




