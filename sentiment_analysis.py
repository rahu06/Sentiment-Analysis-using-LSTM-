###### Requirements######################
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('wordnet')
import numpy as np
nltk.download('punkt')
import pandas as pd
########################################################################################
tokenizer=nltk.RegexpTokenizer(r'\w+')
tokenizer.tokenize('@ hello how are you')
lemmatizer=WordNetLemmatizer()
###########################################################################################
########## Downloading  word Ebeddings ####################################################v
!wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip glove.6B.zip
################################################################################
################### function to convert the downloaded text file into a dictionary of word embeddings ##################################
def add_to_dict(filename):
  words=dict()
  with open(filename,'r')as f:
    for line in f.readlines():
      w =line.split(' ')
      
      words[w[0]]=np.array(w[1:],dtype=float)
      
    return words
##########################################################################################################################################    
######### Words is a dictionary of word embedings######################################    
words=add_to_dict('/content/glove.6B.50d.txt')
#########################################################################################################
################### Method to convert sentences into tokens of valid words that are present in the dictionary############################
def message_to_tokenlist(s):
  tokens= tokenizer.tokenize(s)
  lowercase=[t.lower() for t in tokens]
  lemm=[lemmatizer.lemmatize(t) for t in lowercase]
  useful=[t for t in lemm if  t in words]
  return useful
###################################################################################################################################
################### Method to convert a sentences into feature vectors##################################################
def message_to_word_vectors(message,word_dict):
  processed_listof_tokens=message_to_tokenlist(message)
  print(len(processed_listof_tokens))
  vectors=[]
  for token in processed_listof_tokens:
    if token not in word_dict:
      continue

    token_vector=word_dict[token]
    vectors.append(token_vector)
  return np.array(vectors,dtype=float)


def df_to_x_y(df,words):
  y=df['label'].to_numpy().astype('int')
  all_word_vector_sequences=[]
  for message in df['tweet']:
    print(message)
    message_as_vector_seq=message_to_word_vectors(message,words)
    print(message_as_vector_seq.shape)
    if message_as_vector_seq.shape[0]==0:
      message_as_vector_seq=np.zeros(shape=(1,50))
    all_word_vector_sequences.append(message_as_vector_seq)
  
  return all_word_vector_sequences,y
  
xtrain,ytrain=df_to_x_y(df,words)

##############################################################################################################################################vfrom copy import deepcopy
############################# Method to add padding to sentences to make all senetnces of same size#######################################

def pad(X,desired_sequence_length):
  x_copy=deepcopy(X)
  for i,x in enumerate(X):
    print(x.shape[0])
    x_seq_len=x.shape[0]
    sequence_length_difference=desired_sequence_length-x_seq_len
    pad=np.zeros(shape=(sequence_length_difference,50))
    x_copy[i]=np.concatenate([x,pad])
    print(x_copy[i].shape)
  return np.array(x_copy).astype('float')
    
###############################################################################################
######################### Dataset##############################################
df=pd.DataFrame(
data=[['product is good',1],
      ['product is bad',0],
      ['i am unhappy',0],
      ['unsatisfied',0],
      ['not up to the mark',0],
      ['as expected',1]],columns=['tweet','label'])


#########################################################################################################
################## Building LSTM neural network for Learning##############################################################
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
tf.config.run_functions_eagerly(True)

model=Sequential()
model.add(layers.Input(shape=(7,50)))
model.add(layers.LSTM(64,return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(64,return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(64,return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1,activation='sigmoid'))
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
cp= ModelCheckpoint('model/',save_best_only=True)
model.compile(optimizer=Adam(learning_rate=0.001),loss=BinaryCrossentropy() ,
              metrics=['accuracy'])
model.summary()


x_train=pad(xtrain,7)
x_train.shape
y_train=ytrain.reshape(-1,1)
######################################### Traininig ######################################################
model.fit(x_train,y_train,epochs=2,validation_data=(x_train,y_train),callbacks=[cp])
##################################### Loading the best model ###############################################
from tensorflow.keras.models import load_model
bestmodel=load_model('model/')
prediction=(bestmodel.predict(x_train)>0.5).astype(int)
