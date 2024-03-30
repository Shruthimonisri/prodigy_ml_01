#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


dat = pd.read_csv("C:/Users/91720/OneDrive/Desktop/bharat intern/Housing.csv")


# In[3]:


dat


# In[4]:


dat.head()


# In[5]:


dat.tail()


# In[6]:


dat.shape


# In[7]:


dat.isnull().sum()


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


label_encode = LabelEncoder()


# In[10]:


label = label_encode.fit_transform(dat.mainroad)


# In[11]:


dat['mainroad']= label


# In[12]:


dat


# In[13]:


dat['guestroom']= label


# In[14]:


dat


# In[15]:


dat['furnishingstatus']= label


# In[16]:


dat


# In[17]:


dat['prefarea']= label


# In[18]:


dat


# In[19]:


dat['basement']= label


# In[20]:


dat


# In[21]:


x = dat.drop(['price','basement','hotwaterheating','airconditioning'], axis=1)


# In[22]:


x


# In[23]:


y = dat['price']


# In[24]:


y


# In[25]:


x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state =30)


# In[26]:


x_train.shape


# In[27]:


x_test.shape


# In[28]:


y_train.shape


# In[29]:


y_test.shape


# In[30]:


linear = LinearRegression()


# In[31]:


linear.fit(x_train,y_train)


# In[32]:


linear.score(x_test,y_test)


# In[33]:


from sklearn.ensemble import RandomForestRegressor


# In[34]:


ctf = RandomForestRegressor()


# In[35]:


ctf.fit(x_train,y_train)


# In[36]:


ctf.score(x_test,y_test)


# In[37]:


from sklearn.tree import DecisionTreeRegressor


# In[38]:


tree = DecisionTreeRegressor(random_state = 1)


# In[39]:


tree.fit(x_train,y_train)


# In[40]:


tree.score(x_test,y_test)


# In[41]:


x 


# In[42]:


linear.predict([[7420,4,2,3,1,1,2,1,1]])


# In[43]:


y


# In[44]:


get_ipython().system('pip install tensorflow ')


# In[45]:


from tensorflow import keras


# In[55]:


from tensorflow.keras import layers


# In[47]:


model = keras.Sequential([
     layers.Dense(64, activation='relu', input_shape=[x_train.shape[1]]),
     layers.Dense(32, activation='relu'),
     layers.Dense(1)
    
])


# In[57]:


model.compile(optimizer='rmsprop',loss ='mse', metrics=['mae'])


# In[58]:


history = model.fit(x_train, y_train, epochs=100)


# In[59]:


history = model.fit(x_train, y_train, epochs=100)
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=0)


# In[60]:


loss, mae = model.evaluate(x_test, y_test, verbose=0)
print('Mean Absolute Error:', mae)


# In[ ]:




