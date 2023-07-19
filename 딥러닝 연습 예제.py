#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/user/Desktop/AICE자료/Invistico_Airline.csv")
pd.set_option('display.max_columns',None)
df


# In[3]:


## 정보확인하기
df.info()
'''
22번 columns이 결측치가 존재한다.
'''


# In[4]:


df.describe()


# In[45]:


### 결측치 확인 코드
df.isnull().sum()


# In[54]:


###결측치 치환
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(strategy = 'mean')
df["Arrival Delay in Minutes"] = mean_imputer.fit_transform(df[['Arrival Delay in Minutes']])
df


# In[47]:


df.isnull().sum()
'''
결측치를 평균값으로 치환하였다.
'''


# In[55]:


## object type을 string으로 변환(하는게 좋음)
cols = ['satisfaction','Gender','Customer Type','Type of Travel','Class']
df[cols] = df[cols].astype(str)

##범주형 데이터를 수치형 데이터로 변경하는법 1
df['satisfaction'].replace(['dissatisfied','satisfied'],[0,1],inplace=True)

##순서가 있는 범주형 데이터 손질하기
import pandas as pd

categories = pd.Categorical(df['Class'], 
                            categories = ['Eco','Eco Plus','Business'],
                            ordered = True)
labels,unique = pd.factorize(categories, sort= True)
df['Class'] = labels


# In[57]:


## One hot encoding 하기
cat_cols = ['Gender','Customer Type','Type of Travel']
df = pd.get_dummies(df, columns= cat_cols)
df


# In[62]:


## test set 분리
from sklearn.model_selection import train_test_split
X = df.drop(['satisfaction'],axis=1)
y = df['satisfaction']

X_train, X_test, y_train, y_test=  train_test_split(X,y,test_size=0.3,random_state=42)

## 정규화 진행
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[66]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random

## 모델 시드 고정
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

##Keras의 Sequential 객체로 딥러닝 모델 구성하기
initializer = tf.keras.initializers.GlorotUniform(seed=42) ##모델 시드 고정
model = Sequential()
model.add(Dense(32,activation='relu',input_shape=(25,),kernel_initializer=initializer))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0,3))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0,2))
model.add(Dense(1,activation='sigmoid'))

model.summary()


# In[68]:


# 모델을 학습시킬 최적화 방법, loss함수, 평가 방법 설정
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])


# In[72]:


## 모델 학습하기
es= EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1,restore_best_weights=True)
history = model.fit(X_train,y_train,epochs=100,batch_size=128,
                   verbose=1,validation_data=(X_test,y_test),callbacks=[es])


# In[77]:


###시각화
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'],loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Validation'],loc='upper right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




