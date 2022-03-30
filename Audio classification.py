#!/usr/bin/env python
# coding: utf-8

# # Audio Classification

# #### Installing the librosa library 

# In[33]:


get_ipython().system('pip install librosa')


# ### Performing the EDA on one of the dog sounds available

# In[34]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


filename='UrbanSound8K/UrbanSound8K/UrbanSound8K/audio/fold1/101415-3-0-2.wav'


# In[36]:


import IPython.display as ipd
import librosa
import librosa.display


# In[40]:


### dog bark
import librosa
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename)
librosa.display.waveshow(data,sr=sample_rate)
ipd.Audio(filename)


# ### The sampling rate is nothing but samples taken per second, and by default, librosa samples the file at a sampling rate of 22050

# In[41]:


sample_rate


# In[42]:


from scipy.io import wavfile as wav
wave_sample_rate, wave_audio=wav.read(filename)


# In[43]:


wave_sample_rate


# In[44]:


wave_audio


# #### The data is normalised and ranges between -1 to +1

# In[45]:


data


# In[1]:


import pandas as pd 
import os 
import librosa


# #### Loading the dataset and also the metadata

# In[3]:


audio_dataset_path='UrbanSound8K/UrbanSound8K/UrbanSound8K/audio'
metadata=pd.read_csv('UrbanSound8K/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8k.csv')
metadata.head()


# ### Check whether the dataset is imbalanced

# In[46]:



metadata['class'].value_counts()


# From the above metadata information we can see that the classes of sounds that are engine_idling, dog_bark, street_music, Children_playing, Jackhammer, Drilling, Air_conditioner, siren, Car_horn and gun_shot.

# #### Using the librosa library we are extracting the features

# In[4]:


def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features


# ### Now we iterate through every audio file and extract features 
# ### using Mel-Frequency Cepstral Coefficients

# In[5]:


import numpy as np
from tqdm import tqdm

extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


# ### converting extracted_features to Pandas dataframe

# In[6]:



extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# ### Split the dataset into independent and dependent dataset

# In[7]:



X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[8]:


X.shape


# In[9]:


y


# ### Label Encoding
# ### y=np.array(pd.get_dummies(y))
# ### Label Encoder

# In[10]:



from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[11]:


y


# ### Train Test Split

# In[12]:



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[13]:


X_train


# In[14]:


y


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


y_train.shape


# In[18]:


y_test.shape


# In[19]:


import tensorflow as tf
print(tf.__version__)


# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics


# ### No of classes

# In[53]:



num_labels=y.shape[1]
num_labels


# ### Building the model

# In[22]:


model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[23]:


model.summary()


# In[24]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# ## Trianing my model

# In[58]:



from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 150
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# ## Test Accuracy

# In[59]:


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


# In[60]:


X_test[1]


# In[61]:


predictions = (model.predict(X_test) > 0.5).astype("int32")


# In[62]:


predictions


# #### Example prediction 

# In[63]:


filename="UrbanSound8K/UrbanSound8K/UrbanSound8K/audio/fold1/101415-3-0-2.wav"
prediction_feature=features_extractor(filename)
prediction_feature=prediction_feature.reshape(1,-1)

np.argmax(model.predict(prediction_feature), axis=-1)


# In[64]:


metadata['class'].unique()


# So we can see that the array [1] which is the dog bark is corectly predicted
